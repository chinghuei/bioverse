import torch
from tqdm import tqdm
from pathlib import Path
from clearml import Logger, Task
from .utils import inject_and_tokenize, mask_prompt_loss
from .constants import TRAINABLE_BIO_TOKEN, num_bio_tokens


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

# New alignment training function using InfoNCE loss

def align_model(
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
    learn_rate=2e-5,
    temperature: float = 0.07,
):
    """Contrastively align BIO embeddings with label representations using InfoNCE.

    Parameters
    ----------
    llm_model : PreTrainedModel
        Frozen language model used to contextualise the prompts.
    projection_layer : nn.Module
        Adapter projecting biological embeddings into the LLM space.
    dataloader : DataLoader
        Yields batches of ``(bio_embedding, label)`` tuples.
    tokenizer : PreTrainedTokenizer
        Tokenizer matching ``llm_model``.
    trainable_bio_module : TrainableBIO
        Contains the trainable BIO embedding vector.
    input_text : str
        Prompt template containing ``[BIO_*]`` placeholders.
    device : torch.device
        Computation device.
    logger : Logger
        ClearML logger for metrics.
    task : Task
        ClearML task for artifact tracking.
    checkpoint_dir : Path
        Directory to save intermediate checkpoints.
    epochs : int
        Number of training epochs.
    learn_rate : float
        Optimiser learning rate.
    temperature : float
        Temperature applied to cosine similarities.
    """
    tokenizer.padding_side = "right"
    projection_layer.train()
    trainable_bio_module.train()
    # Freeze LLM parameters during alignment
    llm_model.eval()
    for p in llm_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        list(projection_layer.parameters()) + list(trainable_bio_module.parameters()),
        lr=learn_rate,
    )

    trainable_bio_token_id = tokenizer.convert_tokens_to_ids(TRAINABLE_BIO_TOKEN)
    bio_token_ids = [tokenizer.convert_tokens_to_ids(f"[BIO_{i+1}]") for i in range(num_bio_tokens)]

    for epoch in range(epochs):
        total_loss = 0.0
        for step, (bio_embeddings, labels) in enumerate(tqdm(dataloader, desc=f"Align Epoch {epoch+1}")):
            # Build prompt with injected BIO embeddings
            input_ids, final_embeds, _ = inject_and_tokenize(
                llm_model,
                tokenizer,
                input_text,
                None,
                bio_embeddings,
                projection_layer,
                trainable_bio_module.embedding,
                device,
            )

            outputs = llm_model(
                inputs_embeds=final_embeds,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1]  # B x T x D

            # Pool BIO token representations
            anchors = []
            for i in range(hidden.size(0)):
                mask = torch.zeros_like(input_ids[i], dtype=torch.bool)
                for t_id in bio_token_ids:
                    mask |= input_ids[i] == t_id
                mask |= input_ids[i] == trainable_bio_token_id
                embeds = hidden[i][mask]
                anchors.append(embeds.mean(dim=0))
            anchors = torch.stack(anchors)  # B x D

            # Get label representations from the frozen LLM
            lab_tok = tokenizer(list(labels), return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                lab_outputs = llm_model(
                    input_ids=lab_tok.input_ids,
                    attention_mask=lab_tok.attention_mask,
                    output_hidden_states=True,
                )
            label_hidden = lab_outputs.hidden_states[-1]
            label_mask = lab_tok.attention_mask.unsqueeze(-1)
            label_embeds = (label_hidden * label_mask).sum(dim=1) / label_mask.sum(dim=1)

            # Compute cosine similarity matrix
            norm_anchors = torch.nn.functional.normalize(anchors, dim=1)
            norm_labels = torch.nn.functional.normalize(label_embeds, dim=1)
            logits = torch.matmul(norm_anchors, norm_labels.T) / temperature
            target = torch.arange(logits.size(0), device=device)
            loss = torch.nn.functional.cross_entropy(logits, target)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            logger.report_scalar(
                "Loss",
                "Align_Iteration",
                iteration=epoch * len(dataloader) + step,
                value=loss.item(),
            )

        avg_loss = total_loss / len(dataloader)
        logger.report_scalar("Loss", "Align_Epoch", iteration=epoch, value=avg_loss)
        adapter_path = checkpoint_dir / f"adapter_align_epoch{epoch+1}.pt"
        torch.save(projection_layer.state_dict(), adapter_path)
        task.upload_artifact(name=f"adapter_align_epoch{epoch+1}", artifact_object=str(adapter_path))
        bio_token_path = checkpoint_dir / f"trainable_bio_align_epoch{epoch+1}.pt"
        torch.save(trainable_bio_module.state_dict(), bio_token_path)
        task.upload_artifact(name=f"trainable_bio_align_epoch{epoch+1}", artifact_object=str(bio_token_path))

