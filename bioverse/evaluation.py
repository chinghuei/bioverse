"""Utilities for evaluating a trained Bioverse model."""

from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from clearml import Logger, Task
import torch
from torch import autocast
from .utils import inject_and_tokenize


def evaluate_model(
    llm_model,
    projection_layer,
    dataloader,
    tokenizer,
    trainable_bio_module,
    input_text,
    device,
    logger: Logger,
    task: Task,
):
    """Evaluate the model on a labeled dataset and report metrics."""

    projection_layer.eval()
    llm_model.eval()
    preds, truths = [], []
    tokenizer.padding_side = "left"  # generation models expect left padding
    trainable_bio_token_embedding = trainable_bio_module.embedding
    with torch.no_grad():
        # Iterate over the dataset and generate predictions
        for batch_idx, (bio_embeddings, labels) in enumerate(
            tqdm(dataloader, desc="Evaluating")
        ):
            with autocast(device_type=device.type):
                # Inject the biological embeddings into the prompt
                input_ids, input_embeds, attention_mask = inject_and_tokenize(
                    llm_model,
                    tokenizer,
                    input_text,
                    labels=None,
                    bio_embeddings=bio_embeddings,
                    adapter=projection_layer,
                    trainable_bio_token_embedding=trainable_bio_token_embedding,
                    device=device,
                )
                outputs = llm_model.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    max_length=input_embeds.shape[1] + 20,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
            # Convert generated token IDs back to text and normalise
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_preds = [p.strip().lower() for p in decoded]
            decoded_truths = [t.strip().lower() for t in labels]

            # Print each prediction alongside its ground truth label
            for i, (pred, true) in enumerate(zip(decoded_preds, decoded_truths)):
                print(
                    f"[{batch_idx * dataloader.batch_size + i + 1}] pred: {pred:<30} | truth: {true}"
                )

            preds.extend(decoded_preds)
            truths.extend(decoded_truths)
    # Aggregate predictions to compute overall metrics
    acc = accuracy_score(truths, preds)
    macro_f1 = f1_score(truths, preds, average="macro", zero_division=0)
    print("\nFinal Evaluation Metrics:")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(truths, preds, zero_division=0))
    logger.report_scalar("Accuracy", "Eval", iteration=0, value=acc)
    logger.report_scalar("Macro_F1", "Eval", iteration=0, value=macro_f1)
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
