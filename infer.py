"""Run inference on a subset of the evaluation dataset.

This script loads the trained models and randomly selects several samples from
the dataset. For each sample a prediction is generated and logged via ClearML.
It is primarily meant for quick qualitative checks of the model output.
"""

from datetime import datetime
from pathlib import Path
import random
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
    run_inference,
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
    """Entry point for running ad-hoc inference."""

    # Folder that stores the trained model checkpoints
    checkpoint_dir = Path("checkpoints")

    # Determine computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compose the prompt template used for inference
    bio_tokens = (
        f"{BIO_START_TOKEN} {' '.join(bio_token_list)} {TRAINABLE_BIO_TOKEN} {BIO_END_TOKEN}"
    )
    query = "What cell type is this?"
    prompt_text = f"{query} {bio_tokens} {ANSWER_TOKEN}"

    # ClearML task for experiment tracking
    task = Task.init(
        project_name="MAMMAL-Granite",
        task_name="Infer_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        task_type=Task.TaskTypes.inference,
    )
    logger = Logger.current_logger()

    # Load the fine-tuned model components
    save_model_dir = checkpoint_dir / "final_model"
    (
        llm_model,
        llm_tokenizer,
        b2t_projection_layer,
        trainable_bio_module,
    ) = loadSavedModels(save_model_dir, device)

    # Load the evaluation data. Here a standardised PBMC dataset is used.
    remote_root_data_path = (
        '/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/'
    )
    h5ad_path = (
        remote_root_data_path + '/batch_effect/human_pbmc/h5ad/standardized.h5ad'
    )
    adata = load_AnnData_from_file(h5ad_path, use_subset=False)
    # Prepare a dataset that returns a MAMMAL embedding for each cell
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

    # Randomly sample a subset of cells for demonstration purposes
    random_indices = random.sample(range(len(test_dataset)), 50)
    for idx in random_indices:
        bio_embeddings, label = test_dataset[idx]
        prediction = run_inference(
            llm_model,
            b2t_projection_layer,
            llm_tokenizer,
            trainable_bio_module,
            prompt_text,
            bio_embeddings,
            device,
        )
        logger.report_text(
            f"[{idx}] Predicted: {prediction} | Ground Truth: {label}"
        )


if __name__ == "__main__":
    main()
