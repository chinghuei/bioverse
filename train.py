"""Training script for Bioverse models."""

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
    train_model,
    save_trained_models,
    num_bio_tokens,
    bio_token_list,
    TRAINABLE_BIO_TOKEN,
    BIO_START_TOKEN,
    BIO_END_TOKEN,
    ANSWER_TOKEN,
    special_tokens,
)


def main():
    """Train the LLM on scRNA-seq embeddings."""

    parser = argparse.ArgumentParser(description="Train Bioverse models")
    parser.add_argument(
        "--data",
        type=str,
        default="/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/batch_effect/human_pbmc/h5ad/standardized.h5ad",
        help="Path to an AnnData .h5ad file",
    )
    args = parser.parse_args()

    # Hyperparameters
    use_lora = True
    num_epochs = 1
    batch_size = 32
    learning_rate = 2e-5

    # Prepare checkpoint directory and compute device
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compose the input prompt template
    bio_tokens = (
        f"{BIO_START_TOKEN} {' '.join(bio_token_list)} {TRAINABLE_BIO_TOKEN} {BIO_END_TOKEN}"
    )
    query = "What cell type is this?"
    input_text = f"{query} {bio_tokens} {ANSWER_TOKEN}"

    # Start a ClearML task for experiment management
    task = Task.init(
        project_name="MAMMAL-Granite",
        task_name="Train_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        task_type=Task.TaskTypes.training,
    )
    logger = Logger.current_logger()

    # Record hyperparameters so that the experiment can be reproduced exactly
    task.connect(
        {
            "use_lora": use_lora,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }
    )

    # Load the pre-trained models. ``loadLLM`` optionally wraps the language
    # model with LoRA adapters if ``use_lora`` is True.
    mammal_model, mammal_tokenizer = loadMammal(
        "ibm/biomed.omics.bl.sm.ma-ted-458m", device
    )
    llm_model, llm_tokenizer = loadLLM(
        "ibm-granite/granite-3.3-2b-base", device, use_lora=use_lora
    )
    llm_tokenizer.add_special_tokens(special_tokens)
    llm_model.resize_token_embeddings(len(llm_tokenizer))
    b2t_projection_layer = BioToTextProjectionLayer(num_tokens=num_bio_tokens).to(device)

    # Load the training data from disk. ``args.data`` can be used to override
    # the default PBMC dataset path.
    adata = load_AnnData_from_file(args.data, use_subset=False)

    # Convert gene expression values into embeddings using the MAMMAL encoder
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
    train_dataset, _, _ = split_dataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create a trainable embedding that represents the entire gene expression
    # profile when inserted into the LLM prompt.
    trainable_bio_module = TrainableBIO(llm_model.config.hidden_size).to(device)
    train_model(
        llm_model,
        b2t_projection_layer,
        train_loader,
        llm_tokenizer,
        trainable_bio_module,
        input_text,
        device,
        logger,
        task,
        checkpoint_dir,
        epochs=num_epochs,
        use_lora=use_lora,
        learn_rate=learning_rate,
    )
    # Persist final models to disk for later inference/evaluation
    save_trained_models(
        llm_model,
        llm_tokenizer,
        b2t_projection_layer,
        trainable_bio_module,
        checkpoint_dir,
        use_lora=use_lora,
    )


if __name__ == "__main__":
    main()
