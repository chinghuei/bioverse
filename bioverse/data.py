"""Data loading utilities and dataset wrappers for Bioverse."""

import scanpy as sc
import torch
from torch.utils.data import Dataset, random_split
from mammal.keys import (
    ENCODER_INPUTS_STR,
    ENCODER_INPUTS_TOKENS,
    ENCODER_INPUTS_ATTENTION_MASK,
)
from abc import ABC, abstractmethod
from .constants import num_bio_tokens


def load_AnnData_from_file(h5ad_path, use_subset=False):
    """Load an AnnData object from ``h5ad_path``.

    Parameters
    ----------
    h5ad_path : str or Path
        Location of the ``.h5ad`` file.
    use_subset : bool, optional
        If ``True``, randomly sample 20% of the cells to speed up experiments.

    Returns
    -------
    AnnData
        The loaded dataset (or subset).
    """

    adata_all = sc.read_h5ad(h5ad_path)
    if use_subset:
        adata = adata_all[adata_all.obs.sample(frac=0.2, random_state=42).index, :]
    else:
        adata = adata_all
    return adata


class ScRNASeqEncoder(ABC):
    @abstractmethod
    def get_cell_embedding(self, genes, expressions) -> torch.Tensor:
        pass


class MammalEncoder(ScRNASeqEncoder):
    """Wrapper around the MAMMAL model to produce fixed-length embeddings."""

    def __init__(self, mammal_model, mammal_tokenizer, device, num_tokens):
        self.model = mammal_model
        self.tokenizer = mammal_tokenizer
        self.device = device
        self.num_tokens = num_tokens

    def get_cell_embedding(self, genes, expressions) -> torch.Tensor:
        """Encode a single cell as a fixed-length embedding tensor."""

        # Pick the most highly expressed genes to reduce input length
        top_n = 1024
        sorted_genes = [
            gene for _, gene in sorted(zip(expressions, genes), reverse=True) if _ > 0
        ]
        top_genes_str = "[" + "][".join(sorted_genes[:top_n]) + "]"

        # Construct the string representation expected by the MAMMAL tokenizer
        sample_dict = dict()
        sample_dict[ENCODER_INPUTS_STR] = (
            f"<@TOKENIZER-TYPE=GENE><MOLECULAR_ENTITY><MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>{top_genes_str}<EXPRESSION_DATA_END><EOS>"
        )
        self.tokenizer(
            sample_dict=sample_dict,
            key_in=ENCODER_INPUTS_STR,
            key_out_tokens_ids=ENCODER_INPUTS_TOKENS,
            key_out_attention_mask=ENCODER_INPUTS_ATTENTION_MASK,
        )

        # Run the encoder model to obtain hidden states
        tokens = torch.tensor(sample_dict[ENCODER_INPUTS_TOKENS]).unsqueeze(0).to(self.device)
        attention = torch.tensor(sample_dict[ENCODER_INPUTS_ATTENTION_MASK]).unsqueeze(0).to(self.device)
        batch_dict = {
            ENCODER_INPUTS_TOKENS: tokens,
            ENCODER_INPUTS_ATTENTION_MASK: attention,
            "forward_mode": "encoder",
        }
        with torch.no_grad():
            output = self.model(batch_dict)
            last_hidden = output["model.out.encoder_last_hidden_state"].squeeze(0)

            # Pad or truncate to ``self.num_tokens`` so that downstream modules
            # can assume a fixed size
            if last_hidden.size(0) < self.num_tokens:
                padding = self.num_tokens - last_hidden.size(0)
                pad = torch.zeros(padding, last_hidden.size(1)).to(self.device)
                bio_embeddings = torch.cat([last_hidden, pad], dim=0)[: self.num_tokens]
            else:
                bio_embeddings = last_hidden[: self.num_tokens]

        return bio_embeddings


class AnnDatasetWithBioEmbedding(Dataset):
    """PyTorch ``Dataset`` yielding MAMMAL embeddings and labels."""

    def __init__(self, adata, bio_encoder: ScRNASeqEncoder, device, label_key="CellType"):
        self.adata = adata
        self.labels = adata.obs[label_key].values
        self.genes = adata.var_names.tolist()
        self.bio_encoder = bio_encoder
        self.device = device

    def __len__(self):
        return self.adata.shape[0]

    def __getitem__(self, idx):
        # ``AnnData.X`` may store sparse matrices; convert to a dense vector
        expr = (
            self.adata.X[idx].toarray().flatten()
            if hasattr(self.adata.X[idx], "toarray")
            else self.adata.X[idx]
        )
        # Convert raw gene expression into an embedding using the provided encoder
        bio_embeddings = self.bio_encoder.get_cell_embedding(self.genes, expr)
        return bio_embeddings, self.labels[idx]


from typing import Tuple

def split_dataset(dataset) -> Tuple[Dataset, Dataset, Dataset]:
    """Split a dataset into train/dev/test subsets (60/20/20)."""

    total_size = len(dataset)
    train_size = int(0.6 * total_size)
    dev_size = int(0.2 * total_size)
    test_size = total_size - train_size - dev_size
    train_dataset, dev_dataset, test_dataset = random_split(
        dataset,
        [train_size, dev_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    return train_dataset, dev_dataset, test_dataset
