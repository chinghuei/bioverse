# %% [markdown]
# # BIOVERSE

# %%
from clearml import Task, Logger
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import scanpy as sc
from mammal.model import Mammal
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# %% [markdown]
# ### Define special tokens

# %%
# number of tokens for each bio-entiry. 
# The tokens are usually polled from the last layer of the BioFM
# for a (bio) encoder the first token is usually [CLS]
num_bio_tokens = 2 

# Generate special tokens [BIO_1], [BIO_2], ... 
bio_token_list = [f"[BIO_{i+1}]" for i in range(num_bio_tokens)]    # BIO tokens (static, from BioFM) 

# Define other special tokens
TRAINABLE_BIO_TOKEN = '[TRAINABLE_BIO]' # a trainable token to capture task level attributes
BIO_START_TOKEN = '[BIO_START]' # a operational (not trainable) token to mark the start of BIO
BIO_END_TOKEN = '[BIO_END]' # a operational (not trainable) token to mark the end of BIO
ANSWER_TOKEN = '[ANSWER]' # a operational (not trainable) token to mark the end of the entire prompt

# prepare dictionary of special tokens (for the tokenizer)
special_tokens = {
    'additional_special_tokens': [BIO_START_TOKEN, BIO_END_TOKEN, TRAINABLE_BIO_TOKEN, ANSWER_TOKEN] + bio_token_list
}

# %% [markdown]
# ### Load scRNA-seq data (.h5ad) from file

# %%
def load_AnnData_from_file(h5ad_path, use_subset = False):
    use_subset_ann = use_subset

    adata_all = sc.read_h5ad(h5ad_path)

    if use_subset_ann:
        adata = adata_all[adata_all.obs.sample(frac=0.2, random_state=42).index, :]
    else:
        adata = adata_all

    return adata

# %% [markdown]
# ### Load biomedical foundation model

# %%
# ======== Load MAMMAL Encoder (frozen) ========
def loadMammal(model_path, device):
    mammal_model = Mammal.from_pretrained(model_path).eval().to(device)
    mammal_tokenizer = ModularTokenizerOp.from_pretrained(model_path)
    for p in mammal_model.parameters():
        p.requires_grad = False # Freeze all weights

    return mammal_model, mammal_tokenizer

# %% [markdown]
# ### Load LLM and add optional LoRA

# %%

def loadLLM(model_path, device, use_lora=True):
    # ======== Load Granite (or LLaMA) Decoder ========
    llm_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    llm_tokenizer = AutoTokenizer.from_pretrained(model_path)

    # ========= LoRA ==========
    if use_lora:
        lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
        )
        llm_model = get_peft_model(llm_model, lora_config)
    else:
        # freeze LLM when not using LoRA (this way only the projection layer is updated)
        # do Not freeze LLM when using LoRA, as q_proj & v_proj are still injected into LLM
        for p in llm_model.parameters():
            p.requires_grad = False

    return llm_model, llm_tokenizer

# %% [markdown]
# ### [TRAINABLE_BIO], a single trainable vector

# %%
# A single trainable vector
class TrainableBIO(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(dim))

# %% [markdown]
# ### Define the (trainable) projection layer 

# %%
# ======== Trainable projection layer: Projects Bio embeddings to LLM embedding space ========
class BioToTextProjectionLayer(nn.Module):
    """
    Projects Bio encoder outputs to N token embeddings for LLM input.
    If N=1, this is equivalent to projecting a single [CLS] token, i.e., (B, 1, input_dim)
    If N>1, assume input shape is (B, N, input_dim), 
        e.g., top-N gene tokens when using sorted genes, or any N "feature" tokens 

    input_dim = dimension of bio embeddings (e.g. MAMMAL is 768)
    target_dim = dimension of LLM embeddings (e.g. Granite is 2048)
    """
    def __init__(self, input_dim=768, hidden_dim=1024, target_dim=2048, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens

        # one layer simple projection (as in LLaVA 1)
        # self.proj = nn.Sequential(
        #     nn.Linear(input_dim, target_dim),
        # )

        # two layerer simple MLP (as in LLaVA 2)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim)
        )

    def forward(self, x):
        # Input x shape: (B, input_dim) or (B, N, input_dim)
        if x.ndim == 2:  # (B, input_dim) -> (B, 1, input_dim)
            x = x.unsqueeze(1)
            
        return self.proj(x)  # Output shape: (B, N, target_dim)

# %% [markdown]
# ### Create a PyTorch Dataset specific for AnnData (.h5ad) and use MAMMAL to encode 
# Input: a Anndata object, MAMMAL model and MAMMAL tokenizer, label column name in the Anndata object, number of bio tokens for each cell 
# 
# Output: paired bio embeddings and label text

# %%
from abc import ABC, abstractmethod

# abstract base class (ABC) for scRNA-seq encoder (e.g. MAMMAL, BMFM-RNA)
# an instance of this class is needed to create the Ann dataset, which use the get_cell_embedding() to obtain cell embeddings
class ScRNASeqEncoder(ABC):
    @abstractmethod
    def get_cell_embedding(self, genes, expressions) -> torch.Tensor:
        pass

# %%
from mammal.keys import (
    ENCODER_INPUTS_STR,
    ENCODER_INPUTS_TOKENS,
    ENCODER_INPUTS_ATTENTION_MASK,
)

# MammalEncoder is a subclass of ScRNASeqEncoder
class MammalEncoder(ScRNASeqEncoder):
    def __init__(self, mammal_model, mammal_tokenizer, device, num_tokens):
        self.model = mammal_model
        self.tokenizer = mammal_tokenizer
        self.device = device
        self.num_tokens = num_tokens

    # implementation of the required abstract method (defined in the superclass)
    def get_cell_embedding(self, genes, expressions) -> torch.Tensor:
        top_n = 1024    # Top expressed genes
        sorted_genes = [gene for _, gene in sorted(zip(expressions, genes), reverse=True) if _ > 0]

        # MAMMAL expects something like "[BRCA1][TP53][EGFR]"
        top_genes_str = "[" + "][".join(sorted_genes[:top_n]) + "]"

        # Build MAMMAL input format
        sample_dict = dict()
        sample_dict[ENCODER_INPUTS_STR] = f"<@TOKENIZER-TYPE=GENE><MOLECULAR_ENTITY><MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>{top_genes_str}<EXPRESSION_DATA_END><EOS>"

        # Tokenize for MAMMAL
        self.tokenizer(sample_dict=sample_dict, 
                       key_in=ENCODER_INPUTS_STR,
                       key_out_tokens_ids=ENCODER_INPUTS_TOKENS,
                       key_out_attention_mask=ENCODER_INPUTS_ATTENTION_MASK)
        tokens = torch.tensor(sample_dict[ENCODER_INPUTS_TOKENS]).unsqueeze(0).to(self.device)
        attention = torch.tensor(sample_dict[ENCODER_INPUTS_ATTENTION_MASK]).unsqueeze(0).to(self.device)
        
        batch_dict = {
            ENCODER_INPUTS_TOKENS: tokens,
            ENCODER_INPUTS_ATTENTION_MASK: attention,
            "forward_mode": "encoder"
        }

        with torch.no_grad():
            output = self.model(batch_dict)
            last_hidden = output["model.out.encoder_last_hidden_state"].squeeze(0)
            
            # Get first N (,i.e., num_tokens) token embeddings
            # if N = 1 we got the first token, which is [CLS]

            # --- what we used to do --- 
            # embedding = output["model.out.encoder_last_hidden_state"].mean(dim=1).squeeze(0)
            # --- is not ideal as it's the average of all tokens in the last hidden layerm which also includes [CLS] 
            
            if last_hidden.size(0) < self.num_tokens:
                padding = self.num_tokens - last_hidden.size(0)
                pad = torch.zeros(padding, last_hidden.size(1)).to(self.device)
                bio_embeddings = torch.cat([last_hidden, pad], dim=0)[:self.num_tokens]
            else:
                bio_embeddings = last_hidden[:self.num_tokens]
        return bio_embeddings

# %%
# ------------------------------------------------------------------------------------
# Dataset for AnnData Input and Bio embedding (encoded using the bio_encoder)
# ------------------------------------------------------------------------------------
class AnnDatasetWithBioEmbedding(Dataset):
    def __init__(self, adata, bio_encoder:ScRNASeqEncoder, device, label_key="CellType"):
        self.adata = adata
        self.labels = adata.obs[label_key].values
        self.genes = adata.var_names.tolist()
        self.bio_encoder = bio_encoder
        self.device = device

    def __len__(self):
        return self.adata.shape[0]

    def __getitem__(self, idx):
        expr = self.adata.X[idx].toarray().flatten() if hasattr(self.adata.X[idx], 'toarray') else self.adata.X[idx]
        bio_embeddings = self.get_cell_embeddings(self.genes, expr)

        # bio_embeddings is calculated from self.adata.X[idx]        
        return bio_embeddings, self.labels[idx]
    
    def get_cell_embeddings(self, genes, expr):
        bio_embeddings = self.bio_encoder.get_cell_embedding(genes, expr)
        return bio_embeddings

    # def get_MAMMAL_embeddings(self, genes, expr):
    #     top_n = 1024    # Top expressed genes
    #     sorted_genes = [gene for _, gene in sorted(zip(expr, genes), reverse=True) if _ > 0]

    #     # MAMMAL expects something like "[BRCA1][TP53][EGFR]"
    #     top_genes_str = "[" + "][".join(sorted_genes[:top_n]) + "]"

    #     # Build MAMMAL input format
    #     sample_dict = dict()
    #     sample_dict[ENCODER_INPUTS_STR] = f"<@TOKENIZER-TYPE=GENE><MOLECULAR_ENTITY><MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>{top_genes_str}<EXPRESSION_DATA_END><EOS>"

    #     # Tokenize for MAMMAL
    #     self.mammal_tokenizer(sample_dict=sample_dict,
    #                      key_in=ENCODER_INPUTS_STR,
    #                      key_out_tokens_ids=ENCODER_INPUTS_TOKENS,
    #                      key_out_attention_mask=ENCODER_INPUTS_ATTENTION_MASK)
    #     tokens = torch.tensor(sample_dict[ENCODER_INPUTS_TOKENS]).unsqueeze(0).to(self.device)
    #     attention = torch.tensor(sample_dict[ENCODER_INPUTS_ATTENTION_MASK]).unsqueeze(0).to(self.device)
        
    #     batch_dict = {
    #         ENCODER_INPUTS_TOKENS: tokens,
    #         ENCODER_INPUTS_ATTENTION_MASK: attention,
    #         "forward_mode": "encoder"
    #     }

    #     with torch.no_grad():
    #         output = self.mammal_model(batch_dict)
    #         last_hidden = output["model.out.encoder_last_hidden_state"].squeeze(0)
            
    #         # Get first N (,i.e., num_tokens) token embeddings
    #         # if N = 1 we got the first token, which is [CLS]

    #         # --- what we used to do --- 
    #         # embedding = output["model.out.encoder_last_hidden_state"].mean(dim=1).squeeze(0)
    #         # --- is not ideal as it's the average of all tokens in the last hidden layerm which also includes [CLS] 
            
    #         if last_hidden.size(0) < self.num_tokens:
    #             padding = self.num_tokens - last_hidden.size(0)
    #             pad = torch.zeros(padding, last_hidden.size(1)).to(self.device)
    #             bio_embeddings = torch.cat([last_hidden, pad], dim=0)[:self.num_tokens]
    #         else:
    #             bio_embeddings = last_hidden[:self.num_tokens]
    #     return bio_embeddings


# %% [markdown]
# ### function to turn BIO-related token ids into BIO-related token embeddings
# This is how we inject bio embeddings (from BioFM) into the text input

# %%
def inject_bio_and_trainable_tokens(
    input_ids, input_embeds, bio_embeddings, adapter, trainable_bio_token_embedding, tokenizer
):
    """
    Replaces:
    - [BIO_1] ... [BIO_K] with projected bio embeddings
    - [TRAINABLE_BIO] with a trainable embedding
    Assumes input contains [BIO_START] [BIO_1]...[BIO_K] [TRAINABLE_BIO] [BIO_END]
    
    input_ids: (B, T)
    input_embeds: (B, T, D)
    bio_embeddings: (B, K, D_input)  # K = number of bio tokens
    adapter: projects D_input to D (LLM hidden size)
    trainable_bio_module.embedding: (D,)
    tokenizer: tokenizer with special tokens registered
    """
    # Convert relevant tokens to IDs
    # get [TRAINABLE_BIO] id
    trainable_bio_token_id = tokenizer.convert_tokens_to_ids(TRAINABLE_BIO_TOKEN)

    # Dynamically get all BIO_i token IDs
    num_bio_tokens = bio_embeddings.shape[1]
    bio_token_ids = [tokenizer.convert_tokens_to_ids(f"[BIO_{j+1}]") for j in range(num_bio_tokens)]

    B, T, D = input_embeds.shape
    projected_bio = adapter(bio_embeddings)  # (B, K, D)

    # for each sample i in batch (batch size B)
    for i in range(B):
        # Replace [BIO_j] tokens
        for j, bio_token_id in enumerate(bio_token_ids):
            bio_pos = (input_ids[i] == bio_token_id).nonzero(as_tuple=True)[0]
            if len(bio_pos) > 0:
                input_embeds[i, bio_pos[0]] = projected_bio[i, j]

        # Replace [TRAINABLE_BIO] token
        trainable_pos = (input_ids[i] == trainable_bio_token_id).nonzero(as_tuple=True)[0]
        if len(trainable_pos) > 1:
            raise ValueError(f"Expected at most one {TRAINABLE_BIO_TOKEN} in sample {i}, found {len(trainable_pos)}")
        if len(trainable_pos) == 1:
            input_embeds[i, trainable_pos.item()] = trainable_bio_token_embedding

    return input_embeds


# %% [markdown]
# ### turn everything (including text) from ids to embeddings

# %%
def get_token_embeddings(llm_model, input_ids):
    return llm_model.get_input_embeddings()(input_ids)

def inject_and_tokenize(
    llm_model, tokenizer, prompt_text, labels, bio_embeddings, adapter, trainable_bio_token_embedding, device, max_length=64
):
    """
    Tokenizes [prompt + label], injects projected BIO tokens, and returns token ids and final embeddings.
    """
    B = bio_embeddings.size(0)
    bio_embeddings = bio_embeddings.to(device)

    if labels is not None:
        full_texts = [f"{prompt_text} {label}" for label in labels]
    else:
        full_texts = [prompt_text] * B
        
    tokenized = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)
    embeds = get_token_embeddings(llm_model, input_ids)

    final_embeds = inject_bio_and_trainable_tokens(
        input_ids, embeds, bio_embeddings, adapter, trainable_bio_token_embedding, tokenizer
    )

    return input_ids, final_embeds, attention_mask


# %% [markdown]
# ### function to create an attention mask for training 

# %%
def mask_prompt_loss(input_ids, tokenizer, loss_start_token = ANSWER_TOKEN):
    """
    Masks the prompt portion of input_ids by setting it to -100 (ignored in loss).
    Assumes the target label comes after a special separator token like [ANSWER].

    input_ids: (B, T) tensor
    Returns:
        labels: (B, T) tensor with -100 where loss should not be applied
    """
    labels = input_ids.clone()
    B, T = labels.shape

    # Define the token after which labels begin (you can adjust this)
    # e.g., assume labels come after a custom delimiter like [ANSWER]
    loss_start_id = tokenizer.convert_tokens_to_ids(loss_start_token)

    for i in range(B):
        # Find the index of the loss start token (e.g., [ANSWER])
        start_idx = (input_ids[i] == loss_start_id).nonzero(as_tuple=True)[0]
        if len(start_idx) == 0:
            raise ValueError(f"Could not find loss start token [{loss_start_token}] in input_ids[{i}]")

        # Assume label starts immediately AFTER that token
        start = start_idx.item() + 1
        labels[i, :start] = -100  # mask out everything before the label
    return labels


# %% [markdown]
# ### Training

# %%
# ======== Training Function ========
def train_model(llm_model, projection_layer, dataloader, tokenizer, trainable_bio_module, input_text, device, logger, task, checkpoint_dir, epochs=1, use_lora=True, learn_rate=2e-5):
    # for a decoder model, padding side should be "right" in training and "left" when generating
    tokenizer.padding_side = "right"    # i.e., [PAD] are added at the end

    # always train projection adapter
    projection_layer.train()

    # only train LLM if use LoRA
    llm_model.train() if use_lora else llm_model.eval()

    # What is trainable?
    # 1. weights in the projection layer
    # 2. embedding of the special token [TRAINABLE_BIO] 
    # 3. [optional] LoRA for LLM
    optimizer = torch.optim.AdamW(
        list(projection_layer.parameters()) + list(trainable_bio_module.parameters()) + (list(llm_model.parameters()) if use_lora else []),
        lr=learn_rate
    )

    for epoch in range(epochs):
        total_loss = 0
        for step, (bio_embeddings, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            input_ids, final_embeds, attention_mask = inject_and_tokenize(
                llm_model, tokenizer, input_text, labels, bio_embeddings, projection_layer, trainable_bio_module.embedding, device
            )
            labels_full = mask_prompt_loss(input_ids, tokenizer)

            outputs = llm_model(inputs_embeds=final_embeds, labels=labels_full)
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            logger.report_scalar("Loss", "Train_Iteration", iteration=epoch * len(dataloader) + step, value=loss.item())

        avg_loss = total_loss / len(dataloader)
        logger.report_scalar("Loss", "Train_Epoch", iteration=epoch, value=avg_loss)

        # Save weights (per epoch)
        adapter_path = checkpoint_dir / f"adapter_epoch{epoch+1}.pt"
        torch.save(projection_layer.state_dict(), adapter_path)
        task.upload_artifact(name=f"adapter_epoch{epoch+1}", artifact_object=str(adapter_path))
        if use_lora:
            llm_model.save_pretrained(checkpoint_dir / f"llm_epoch{epoch+1}")


# %% [markdown]
# ### Evaluation

# %%
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.amp import autocast
import numpy as np

def evaluate_model(llm_model, projection_layer, dataloader, tokenizer, trainable_bio_module, input_text, device, logger, task):
    projection_layer.eval()
    llm_model.eval()
    preds, truths = [], []

    tokenizer.padding_side = "left"

    # retrive the already trained [TRAINABLE_BIO] embedding from LLM model
    trainable_bio_token_id = tokenizer.convert_tokens_to_ids(TRAINABLE_BIO_TOKEN)
 #   trainable_bio_token_embedding = TrainableBIO(llm_model.config.hidden_size).to(device).embedding
 #   trainable_bio_token_embedding = llm_model.get_input_embeddings().weight[trainable_bio_token_id].to(device)
    trainable_bio_token_embedding = trainable_bio_module.embedding  
    
    with torch.no_grad():
        for batch_idx, (bio_embeddings, labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
            with autocast(device_type=device.type):  # Optional: memory saving
                input_ids, input_embeds, attention_mask = inject_and_tokenize(
                    llm_model, tokenizer, input_text, 
                    labels=None,  # <-- no labels needed (for generation)
                    bio_embeddings=bio_embeddings,
                    adapter=projection_layer,
                    trainable_bio_token_embedding = trainable_bio_token_embedding,
                    device=device
                )

                outputs = llm_model.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    max_length=input_embeds.shape[1] + 20,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False
                )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_preds = [p.strip().lower() for p in decoded]
            decoded_truths = [t.strip().lower() for t in labels]

            for i, (pred, true) in enumerate(zip(decoded_preds, decoded_truths)):
                print(f"[{batch_idx * dataloader.batch_size + i + 1}] pred: {pred:<30} | truth: {true}")

            preds.extend(decoded_preds)
            truths.extend(decoded_truths)

    # === Metrics ===
    acc = accuracy_score(truths, preds)
    macro_f1 = f1_score(truths, preds, average='macro', zero_division=0)

    print("\nFinal Evaluation Metrics:")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(truths, preds, zero_division=0))

    logger.report_scalar("Accuracy", "Eval", iteration=0, value=acc)
    logger.report_scalar("Macro_F1", "Eval", iteration=0, value=macro_f1)

    # confusion matrix
    try:
        from sklearn.metrics import ConfusionMatrixDisplay
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 10))
        ConfusionMatrixDisplay.from_predictions(truths, preds, ax=ax, xticks_rotation=45)
        plt.title("Confusion Matrix")
        task.logger.report_matplotlib_figure("ConfusionMatrix", "Eval", iteration=0, figure=fig)
        plt.show()
    except Exception as e:
        print(f"Could not display confusion matrix: {e}")


# %% [markdown]
# ### Inference (using a single prompt)

# %%
def run_inference(llm_model, adapter, tokenizer, trainable_bio_module, prompt_text, mammal_embedding, device):
    llm_model.eval()
    adapter.eval()

    with torch.no_grad():
        bio_embeddings = mammal_embedding.unsqueeze(0).to(device)  # (1, N, D)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        prompt_embeds = get_token_embeddings(llm_model, prompt_ids)

        # retrive the already trained [TRAINABLE_BIO] embedding from LLM model
        trainable_bio_token_id = tokenizer.convert_tokens_to_ids(TRAINABLE_BIO_TOKEN)
        #trainable_bio_token_embedding = llm_model.get_input_embeddings().weight[trainable_bio_token_id] 
        trainable_bio_token_embedding = trainable_bio_module.embedding  

        input_embeds = inject_bio_and_trainable_tokens(
            prompt_ids, prompt_embeds, bio_embeddings, adapter, trainable_bio_token_embedding, tokenizer
        )

        # only inputs_embeds is needed (i.e. no inputs_ids) 
        # because we are doing 1 sample at a time (i.e. no batching) and there is no padding, 
        # and no need for attention_mask (all tokens are real and assumes full attention)
        generated_ids = llm_model.generate(
            inputs_embeds=input_embeds,
            max_length=input_embeds.shape[1] + 20,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

        return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

# %% [markdown]
# ### Simple helper functions

# %%
from typing import Tuple

# split a PyTorch Dataset into train/dev/test
def split_dataset(dataset) -> Tuple[Dataset, Dataset, Dataset]:
    # Define split sizes
    total_size = len(dataset)
    train_size = int(0.6 * total_size)
    dev_size = int(0.2 * total_size)
    test_size = total_size - train_size - dev_size  # to catch rounding errors

    # Random split
    train_dataset, dev_dataset, test_dataset = random_split(
        dataset, 
        [train_size, dev_size, test_size],
        generator=torch.Generator().manual_seed(42)  # reproducible split
    )

    return train_dataset, dev_dataset, test_dataset

# %%
def save_trained_models(llm_model, llm_tokenizer, b2t_projection_layer, trainable_bio_module, checkpoint_dir, use_lora=True):
    # Final save directory
    final_model_dir = checkpoint_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)

    # Save projection layer
    adapter_path = final_model_dir / "mammal_to_llm_adapter.pt"
    torch.save(b2t_projection_layer.state_dict(), adapter_path)
    #task.upload_artifact(name="final_adapter", artifact_object=str(adapter_path))

    # Resize and make sure config reflects it (added [BIO_j] and other special tokens)
    llm_model.config.vocab_size = len(llm_tokenizer)  # e.g., 49154 with added special tokens
    llm_model.resize_token_embeddings(len(llm_tokenizer))  # Adjust the embedding layer

    # Save LLM (optionally with LoRA)
    if use_lora:
        llm_model_merged_with_lora = llm_model.merge_and_unload()
        llm_model_merged_with_lora.resize_token_embeddings(len(llm_tokenizer))

        llm_model_merged_with_lora.save_pretrained(final_model_dir / "granite_lora")
        del llm_model_merged_with_lora
    #    task.upload_artifact(name="final_lora_llm", artifact_object=str(final_model_dir / "granite_lora"))
    else:
        # Save model and config for non-LoRA zero-shot inference
        llm_model.save_pretrained(final_model_dir / "granite")
    #    task.upload_artifact(name="final_llm", artifact_object=str(final_model_dir / "granite"))

    # Save tokenizer
    llm_tokenizer.save_pretrained(final_model_dir / "tokenizer")
    #task.upload_artifact(name="llm_tokenizer", artifact_object=str(final_model_dir / "tokenizer"))

    # Save trainable [TRAINABLE_BIO] embedding
    bio_token_path = final_model_dir / "trainable_bio_embedding.pt"
    torch.save(trainable_bio_module.state_dict(), bio_token_path)

# %%
def loadSavedModels(save_model_dir, device):
    # === Load the tokenizer first ===
    llm_tokenizer = AutoTokenizer.from_pretrained(save_model_dir / "tokenizer")
#    llm_tokenizer.add_special_tokens(special_tokens)
    llm_tokenizer.padding_side = "left" # default in inference mode, [PAD]s are added to the beginning 

    # == Load the config.json from saved model ===
    # This is important as we added new special tokens and 
    #    the # of tokens in the new model (e.g. 49,154) <--- saved in the model’s embedding.weight (loaded from checkpoint)
    #    the # of tokens in the base model (e.g. 49,152) <--- model’s config (from config.json): says vocab_size = 49152 (without special tokens)
    # config.json should be saved if we 
    #   1. do not use LoRA
    #   2. use LoRA and merge it back to the base
    # This is because Hugging Face does not create a new config.json for PEFT adapters like LoRA
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(save_model_dir / "granite_lora")

    # === Load final saved model ===
    llm_model = AutoModelForCausalLM.from_pretrained(
        save_model_dir / "granite_lora",
        config=config).to(device)
    llm_model.resize_token_embeddings(len(llm_tokenizer))
    
    b2t_projection_layer = BioToTextProjectionLayer(num_tokens=num_bio_tokens).to(device)
    b2t_projection_layer.load_state_dict(torch.load(save_model_dir / "mammal_to_llm_adapter.pt"))
    b2t_projection_layer.eval()

    trainable_bio_module = TrainableBIO(llm_model.config.hidden_size).to(device)
    trainable_bio_module.load_state_dict(torch.load(save_model_dir / "trainable_bio_embedding.pt"))
    trainable_bio_module.eval()

    return llm_model, llm_tokenizer, b2t_projection_layer, trainable_bio_module

# %% [markdown]
# ## Main function to run everything

# %%
# ========== Configuration ==========
use_lora = True

num_epochs = 1
batch_size = 32

learning_rate = 2e-5

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

# ========== Device Configuration ==========
#device = "mps"  # for Mac
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Example input with ground-truth label
# "what is the most likely cell types from [BIO_START] [BIO_1] [BIO_2] [BIO_3] [TRAINABLE_BIO] [BIO_END]? [ANSWER] CD4 T Cell"

# Create the tokenized BIO section
bio_tokens = f"{BIO_START_TOKEN} {' '.join(bio_token_list)} {TRAINABLE_BIO_TOKEN} {BIO_END_TOKEN}"

query = "What cell type is this?"
input_text = f"{query} {bio_tokens} {ANSWER_TOKEN}"

print(input_text)
print(special_tokens)

# %% [markdown]
# ### ClearML Init

# %%
# ========= ClearML Init ==========
task = Task.init(
    project_name = "MAMMAL-Granite",
    task_name = "[Interactive] Zero-Shot_Cell_Type_Annotation_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
    task_type= Task.TaskTypes.training
)
logger = Logger.current_logger()

task.connect({
    "use_lora": use_lora,
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate
})

# %%
# ============================================================================================================
# Models
# ============================================================================================================

# 1. Biomedical Foundation Model
mammal_model, mammal_tokenizer = loadMammal("ibm/biomed.omics.bl.sm.ma-ted-458m", device)
##/dccstor/bmfm-targets/models/omics/transcriptome/scRNA/pretrain/bmfm.omics.bert.110m.scRNA.multitask.v3

# 2. LLM, optinally add a LoRA layer
llm_model, llm_tokenizer = loadLLM("ibm-granite/granite-3.3-2b-base", device, use_lora=use_lora)

# --- Inject special tokens to the LLM tokenizer before tokenization ---
prev_num_tokens = len(llm_tokenizer)
llm_tokenizer.add_special_tokens(special_tokens)

current_num_tokens = len(llm_tokenizer)
llm_model.resize_token_embeddings(current_num_tokens)

print(f"Number of tokens incresed from {prev_num_tokens} to {current_num_tokens}")
print(f"Special tokens added: {special_tokens}")

# 3. Projection layer 
# instantiate a project layer using the class
b2t_projection_layer = BioToTextProjectionLayer(num_tokens = num_bio_tokens).to(device)

# %%

# ============================================================================================================
# Data
# ============================================================================================================

# 1. Load scRNA-seq data from file
# human PBMC cell type classification task from scEval
remote_root_data_path = '/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/'
h5ad_path = remote_root_data_path + '/batch_effect/human_pbmc/h5ad/standardized.h5ad'
adata = load_AnnData_from_file(h5ad_path, use_subset=False)

# 2. Create a PyTouch DataSet object from anndata
mammal_encoder = MammalEncoder(mammal_model, mammal_tokenizer, device, num_bio_tokens) 
dataset = AnnDatasetWithBioEmbedding(adata, mammal_encoder, device, label_key="CellType")

### dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 3. Split dataset into train/dev/test
train_dataset, dev_dataset, test_dataset = split_dataset(dataset)

# 4. Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# %%

# ======== RUN EVERYTHING ========
run_training_mode = True
run_evaluation_mode = True
run_inference_mode = True

# ============================================================================================================
# Training and Evaluating
# ============================================================================================================

# 5. Training
if run_training_mode:  
    # Trainable BIO token [BIO_TRAINABLE]
    # create a new / randomly initialized token for training
    trainable_bio_module = TrainableBIO(llm_model.config.hidden_size).to(device)

    train_model(llm_model, b2t_projection_layer, train_loader, llm_tokenizer, trainable_bio_module, input_text, device, logger, task, checkpoint_dir, epochs=num_epochs, use_lora=use_lora, learn_rate=learning_rate)
    save_trained_models(llm_model, llm_tokenizer, b2t_projection_layer, trainable_bio_module, checkpoint_dir, use_lora=use_lora)

# 6. Evaluation
if run_evaluation_mode:
    save_model_dir = checkpoint_dir / "final_model"
    llm_model, llm_tokenizer, b2t_projection_layer, trainable_bio_module = loadSavedModels(save_model_dir, device)
    evaluate_model(llm_model, b2t_projection_layer, test_loader, llm_tokenizer, trainable_bio_module, input_text, device, logger, task)

    # del test_loader
    # gc.collect()
    # torch.cuda.empty_cache()


# %%

# # Delete variables
# del llm_model_merged_with_lora  # del model

# # Collect garbage
gc.collect()

# # Release unreferenced memory held by PyTorch
torch.cuda.empty_cache()

# # # Clear lingering variables from earlier cells (only in notebook)
# # %reset -f

# %%
print(torch.cuda.memory_summary())

try:
    torch.empty((1024, 1024, 1024), device='cuda')  # ~4 GB
except RuntimeError as e:
    print("OOM likely due to fragmentation:", e)

# %%
if run_inference_mode:
    save_model_dir = checkpoint_dir / "final_model"
    llm_model, llm_tokenizer, b2t_projection_layer, trainable_bio_module = loadSavedModels(save_model_dir, device)

    logger = Logger.current_logger()

    query = "What cell type is this?"
    #query = "What is the most likely cell type?"
    prompt_text = f"{query} {bio_tokens} {ANSWER_TOKEN}"

 #   prompt_text = "What cell type is this? [BIO_START] [BIO_1] [BIO_2] [BIO_END] [ANSWER]"

    print(prompt_text)

    # === Get 50 random cells from AnnData ===
    import random

    random_indices = random.sample(range(len(test_dataset)), 50)

    for idx in random_indices:
        bio_embeddings, label = test_dataset[idx]

        # === Run inference ===
        prediction = run_inference(llm_model, b2t_projection_layer, llm_tokenizer, trainable_bio_module, prompt_text, bio_embeddings, device)

        # === Log to ClearML ===
        logger.report_text(f"[{idx}] Predicted: {prediction} | Ground Truth: {label}")


