"""Model loading and architecture components used in Bioverse."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from mammal.model import Mammal
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
from .constants import num_bio_tokens


def loadMammal(model_path, device):
    """Load the MAMMAL encoder model and tokenizer."""

    mammal_model = Mammal.from_pretrained(model_path).eval().to(device)
    mammal_tokenizer = ModularTokenizerOp.from_pretrained(model_path)
    for p in mammal_model.parameters():
        p.requires_grad = False
    return mammal_model, mammal_tokenizer


def loadLLM(model_path, device, use_lora=True):
    """Load the base language model and optionally apply LoRA adapters."""

    llm_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
    if use_lora:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        llm_model = get_peft_model(llm_model, lora_config)
    else:
        for p in llm_model.parameters():
            p.requires_grad = False
    return llm_model, llm_tokenizer


class TrainableBIO(nn.Module):
    """Simple module containing a single trainable embedding vector."""

    def __init__(self, dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(dim))


class BioToTextProjectionLayer(nn.Module):
    """Project MAMMAL embeddings into the LLM embedding space."""

    def __init__(self, input_dim=768, hidden_dim=1024, target_dim=2048, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, x):
        if x.ndim == 2:
            # When a single sample is provided, add a batch dimension
            x = x.unsqueeze(0)
        # ``self.proj`` processes each token embedding independently
        return self.proj(x)
