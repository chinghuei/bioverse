"""Evaluation script for the Bioverse models.

This module loads the pre-trained models and runs them on a held-out test set in
order to compute various metrics. The evaluation routine largely mirrors the
training and inference pipelines but disables gradient computation and performs
no parameter updates.
"""

from datetime import datetime
from pathlib import Path
import argparse
import torch
from clearml import Task, Logger
from torch.utils.data import DataLoader
from bioverse import (
    loadMammal,
    loadLLM,
    BioToTextProjectionLayer,
    TrainableBIO,
    load_AnnData_from_file,
    MammalEncoder,
    AnnDatasetWithBioEmbedding,
    split_dataset,
    evaluate_model,
    loadSavedModels,
    num_bio_tokens,
    bio_token_list,
    TRAINABLE_BIO_TOKEN,
    BIO_START_TOKEN,
    BIO_END_TOKEN,
    ANSWER_TOKEN,
    special_tokens,
)


def main():
    """Entry point for running model evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate Bioverse models")
    parser.add_argument(
        "--data",
        type=str,
        default="/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/batch_effect/human_pbmc/h5ad/standardized.h5ad",
        help="Path to an AnnData .h5ad file",
    )
    args = parser.parse_args()

    # Directory containing the trained model artifacts
    checkpoint_dir = Path("checkpoints")

    # Select the appropriate device; prefer GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compose the full prompt that will be fed to the LLM. The tokens between
    # ``BIO_START_TOKEN`` and ``BIO_END_TOKEN`` will be replaced by embeddings of
    # biological expression data at runtime.
    bio_tokens = (
        f"{BIO_START_TOKEN} {' '.join(bio_token_list)} {TRAINABLE_BIO_TOKEN} {BIO_END_TOKEN}"
    )
    query = "What cell type is this?"
    input_text = f"{query} {bio_tokens} {ANSWER_TOKEN}"

    # Initialize a ClearML task for experiment tracking. This allows logging of
    # metrics and artifacts in a reproducible manner.
    task = Task.init(
        project_name="MAMMAL-Granite",
        task_name="Eval_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        task_type=Task.TaskTypes.testing,
    )
    logger = Logger.current_logger()

    # Load the trained language model, tokenizer and projection layers from disk
    save_model_dir = checkpoint_dir / "final_model"
    (
        llm_model,
        llm_tokenizer,
        b2t_projection_layer,
        trainable_bio_module,
    ) = loadSavedModels(save_model_dir, device)

    # Path to the annotated data matrix (AnnData) used for evaluation. Only a
    # standardised human PBMC dataset is used here but this could be replaced
    # with any compatible dataset.
    # Load the evaluation data. ``args.data`` can point to any compatible AnnData file.
    adata = load_AnnData_from_file(args.data, use_subset=False)
    # Load the MAMMAL encoder and create a dataset that yields a MAMMAL
    # embedding for each cell alongside its ground-truth label.
    mammal_model, mammal_tokenizer = loadMammal(
        "ibm/biomed.omics.bl.sm.ma-ted-458m", device
    )
    mammal_encoder = MammalEncoder(
        mammal_model,
        mammal_tokenizer,
        device,
        num_bio_tokens,
    )
    dataset = AnnDatasetWithBioEmbedding(
        adata,
        mammal_encoder,
        device,
        label_key="CellType",
    )
    _, _, test_dataset = split_dataset(dataset)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # Run evaluation which prints metrics and logs them to ClearML.
    evaluate_model(
        llm_model,
        b2t_projection_layer,
        test_loader,
        llm_tokenizer,
        trainable_bio_module,
        input_text,
        device,
        logger,
        task,
    )


if __name__ == "__main__":
    main()
