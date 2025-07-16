import torch
from .utils import get_token_embeddings, inject_bio_and_trainable_tokens


def run_inference(
    llm_model,
    adapter,
    tokenizer,
    trainable_bio_module,
    prompt_text,
    mammal_embedding,
    device,
):
    llm_model.eval()
    adapter.eval()
    with torch.no_grad():
        bio_embeddings = mammal_embedding.unsqueeze(0).to(device)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        prompt_embeds = get_token_embeddings(llm_model, prompt_ids)
        trainable_bio_token_embedding = trainable_bio_module.embedding
        input_embeds = inject_bio_and_trainable_tokens(
            prompt_ids,
            prompt_embeds,
            bio_embeddings,
            adapter,
            trainable_bio_token_embedding,
            tokenizer,
        )
        generated_ids = llm_model.generate(
            inputs_embeds=input_embeds,
            max_length=input_embeds.shape[1] + 20,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
