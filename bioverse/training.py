import torch
from tqdm import tqdm
from pathlib import Path
from clearml import Logger, Task
from .utils import inject_and_tokenize, mask_prompt_loss


def train_model(
    llm_model,
    projection_layer,
    dataloader,
    tokenizer,
    trainable_bio_module,
    input_text,
    device,
    logger: Logger,
    task: Task,
    checkpoint_dir: Path,
    epochs=1,
    use_lora=True,
    learn_rate=2e-5,
):
    tokenizer.padding_side = "right"
    projection_layer.train()
    llm_model.train() if use_lora else llm_model.eval()
    optimizer = torch.optim.AdamW(
        list(projection_layer.parameters())
        + list(trainable_bio_module.parameters())
        + (list(llm_model.parameters()) if use_lora else []),
        lr=learn_rate,
    )
    for epoch in range(epochs):
        total_loss = 0
        for step, (bio_embeddings, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            input_ids, final_embeds, _ = inject_and_tokenize(
                llm_model,
                tokenizer,
                input_text,
                labels,
                bio_embeddings,
                projection_layer,
                trainable_bio_module.embedding,
                device,
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
        adapter_path = checkpoint_dir / f"adapter_epoch{epoch+1}.pt"
        torch.save(projection_layer.state_dict(), adapter_path)
        task.upload_artifact(name=f"adapter_epoch{epoch+1}", artifact_object=str(adapter_path))
        if use_lora:
            llm_model.save_pretrained(checkpoint_dir / f"llm_epoch{epoch+1}")

