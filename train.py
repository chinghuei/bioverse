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
    use_lora = True
    num_epochs = 1
    batch_size = 32
    learning_rate = 2e-5
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bio_tokens = f"{BIO_START_TOKEN} {' '.join(bio_token_list)} {TRAINABLE_BIO_TOKEN} {BIO_END_TOKEN}"
    query = "What cell type is this?"
    input_text = f"{query} {bio_tokens} {ANSWER_TOKEN}"

    task = Task.init(
        project_name="MAMMAL-Granite",
        task_name="Train_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        task_type=Task.TaskTypes.training,
    )
    logger = Logger.current_logger()
    task.connect(
        {
            "use_lora": use_lora,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }
    )

    mammal_model, mammal_tokenizer = loadMammal("ibm/biomed.omics.bl.sm.ma-ted-458m", device)
    llm_model, llm_tokenizer = loadLLM("ibm-granite/granite-3.3-2b-base", device, use_lora=use_lora)
    llm_tokenizer.add_special_tokens(special_tokens)
    llm_model.resize_token_embeddings(len(llm_tokenizer))
    b2t_projection_layer = BioToTextProjectionLayer(num_tokens=num_bio_tokens).to(device)

    remote_root_data_path = '/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/'
    h5ad_path = remote_root_data_path + '/batch_effect/human_pbmc/h5ad/standardized.h5ad'
    adata = load_AnnData_from_file(h5ad_path, use_subset=False)

    mammal_encoder = MammalEncoder(mammal_model, mammal_tokenizer, device, num_bio_tokens)
    dataset = AnnDatasetWithBioEmbedding(adata, mammal_encoder, device, label_key="CellType")
    train_dataset, _, _ = split_dataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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
