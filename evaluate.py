from datetime import datetime
from pathlib import Path
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
    checkpoint_dir = Path("checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bio_tokens = f"{BIO_START_TOKEN} {' '.join(bio_token_list)} {TRAINABLE_BIO_TOKEN} {BIO_END_TOKEN}"
    query = "What cell type is this?"
    input_text = f"{query} {bio_tokens} {ANSWER_TOKEN}"

    task = Task.init(
        project_name="MAMMAL-Granite",
        task_name="Eval_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        task_type=Task.TaskTypes.testing,
    )
    logger = Logger.current_logger()

    save_model_dir = checkpoint_dir / "final_model"
    llm_model, llm_tokenizer, b2t_projection_layer, trainable_bio_module = loadSavedModels(save_model_dir, device)

    remote_root_data_path = '/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/'
    h5ad_path = remote_root_data_path + '/batch_effect/human_pbmc/h5ad/standardized.h5ad'
    adata = load_AnnData_from_file(h5ad_path, use_subset=False)
    mammal_model, mammal_tokenizer = loadMammal("ibm/biomed.omics.bl.sm.ma-ted-458m", device)
    mammal_encoder = MammalEncoder(mammal_model, mammal_tokenizer, device, num_bio_tokens)
    dataset = AnnDatasetWithBioEmbedding(adata, mammal_encoder, device, label_key="CellType")
    _, _, test_dataset = split_dataset(dataset)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
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
