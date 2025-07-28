"""Utility functions for embedding injection and model saving/loading."""

import torch
from .constants import (
    TRAINABLE_BIO_TOKEN,
    ANSWER_TOKEN,
)


def get_token_embeddings(llm_model, input_ids):
    """Fetch the embedding vectors corresponding to ``input_ids``."""

    return llm_model.get_input_embeddings()(input_ids)


def inject_bio_and_trainable_tokens(
    input_ids,
    input_embeds,
    bio_embeddings,
    adapter,
    trainable_bio_token_embedding,
    tokenizer,
):
    """Inject biological and trainable token embeddings into a prompt."""

    trainable_bio_token_id = tokenizer.convert_tokens_to_ids(TRAINABLE_BIO_TOKEN)
    num_bio_tokens = bio_embeddings.shape[1]
    bio_token_ids = [tokenizer.convert_tokens_to_ids(f"[BIO_{j+1}]") for j in range(num_bio_tokens)]
    B, T, D = input_embeds.shape
    projected_bio = adapter(bio_embeddings)
    for i in range(B):
        # Replace placeholder BIO tokens with projected embeddings
        for j, bio_token_id in enumerate(bio_token_ids):
            bio_pos = (input_ids[i] == bio_token_id).nonzero(as_tuple=True)[0]
            if len(bio_pos) > 0:
                input_embeds[i, bio_pos[0]] = projected_bio[i, j]

        # Optionally replace the TRAINABLE_BIO token if present
        trainable_pos = (input_ids[i] == trainable_bio_token_id).nonzero(as_tuple=True)[0]
        if len(trainable_pos) > 1:
            raise ValueError(
                f"Expected at most one {TRAINABLE_BIO_TOKEN} in sample {i}, found {len(trainable_pos)}"
            )
        if len(trainable_pos) == 1:
            input_embeds[i, trainable_pos.item()] = trainable_bio_token_embedding
    return input_embeds


def inject_and_tokenize(
    llm_model,
    tokenizer,
    prompt_text,
    labels,
    bio_embeddings,
    adapter,
    trainable_bio_token_embedding,
    device,
    max_length=64,
):
    """Tokenize prompts and inject biological embeddings into them."""

    B = bio_embeddings.size(0)
    bio_embeddings = bio_embeddings.to(device)

    # Build the full prompt for each sample
    if labels is not None:
        full_texts = [f"{prompt_text} {label}" for label in labels]
    else:
        full_texts = [prompt_text] * B
    # Tokenize using the LLM tokenizer
    tokenized = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)
    embeds = get_token_embeddings(llm_model, input_ids)
    final_embeds = inject_bio_and_trainable_tokens(
        input_ids,
        embeds,
        bio_embeddings,
        adapter,
        trainable_bio_token_embedding,
        tokenizer,
    )
    return input_ids, final_embeds, attention_mask


def mask_prompt_loss(input_ids, tokenizer, loss_start_token=ANSWER_TOKEN):
    """Mask tokens so that loss is only computed on the answer portion."""

    labels = input_ids.clone()
    B, T = labels.shape
    loss_start_id = tokenizer.convert_tokens_to_ids(loss_start_token)
    for i in range(B):
        start_idx = (input_ids[i] == loss_start_id).nonzero(as_tuple=True)[0]
        if len(start_idx) == 0:
            raise ValueError(f"Could not find loss start token [{loss_start_token}] in input_ids[{i}]")
        start = start_idx.item() + 1
        labels[i, :start] = -100
    return labels


def save_trained_models(llm_model, llm_tokenizer, b2t_projection_layer, trainable_bio_module, checkpoint_dir, use_lora=True):
    """Persist the fine-tuned model components to ``checkpoint_dir``."""

    final_model_dir = checkpoint_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = final_model_dir / "mammal_to_llm_adapter.pt"
    torch.save(b2t_projection_layer.state_dict(), adapter_path)
    llm_model.config.vocab_size = len(llm_tokenizer)
    llm_model.resize_token_embeddings(len(llm_tokenizer))
    if use_lora:
        llm_model_merged_with_lora = llm_model.merge_and_unload()
        llm_model_merged_with_lora.resize_token_embeddings(len(llm_tokenizer))
        llm_model_merged_with_lora.save_pretrained(final_model_dir / "granite_lora")
        del llm_model_merged_with_lora
    else:
        llm_model.save_pretrained(final_model_dir / "granite")
    llm_tokenizer.save_pretrained(final_model_dir / "tokenizer")
    bio_token_path = final_model_dir / "trainable_bio_embedding.pt"
    torch.save(trainable_bio_module.state_dict(), bio_token_path)


def loadSavedModels(save_model_dir, device):
    """Load model components previously saved with :func:`save_trained_models`."""

    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    from .models import BioToTextProjectionLayer, TrainableBIO
    from .constants import num_bio_tokens
    llm_tokenizer = AutoTokenizer.from_pretrained(save_model_dir / "tokenizer")
    llm_tokenizer.padding_side = "left"
    config = AutoConfig.from_pretrained(save_model_dir / "granite_lora")
    llm_model = AutoModelForCausalLM.from_pretrained(save_model_dir / "granite_lora", config=config).to(device)
    llm_model.resize_token_embeddings(len(llm_tokenizer))
    b2t_projection_layer = BioToTextProjectionLayer(num_tokens=num_bio_tokens).to(device)
    b2t_projection_layer.load_state_dict(torch.load(save_model_dir / "mammal_to_llm_adapter.pt"))
    b2t_projection_layer.eval()
    trainable_bio_module = TrainableBIO(llm_model.config.hidden_size).to(device)
    trainable_bio_module.load_state_dict(torch.load(save_model_dir / "trainable_bio_embedding.pt"))
    trainable_bio_module.eval()
    return llm_model, llm_tokenizer, b2t_projection_layer, trainable_bio_module
