from pathlib import Path
from typing import List

import numpy as np
import torch
import scanpy as sc
from fastapi import FastAPI
from pydantic import BaseModel

from bioverse import (
    loadMammal,
    MammalEncoder,
    num_bio_tokens,
    loadSavedModels,
    run_inference,
    BIO_START_TOKEN,
    BIO_END_TOKEN,
    TRAINABLE_BIO_TOKEN,
    ANSWER_TOKEN,
    bio_token_list,
)

app = FastAPI()

# Load dataset and models once at startup
adata = sc.datasets.pbmc68k_reduced()
if "X_umap" not in adata.obsm:
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adjust the checkpoint path as needed
try:
    llm_model, tokenizer, adapter, trainable_bio = loadSavedModels(
        Path("checkpoints/final_model"), device
    )
    mammal_model, mammal_tokenizer = loadMammal(
        "ibm/biomed.omics.bl.sm.ma-ted-458m", device
    )
    encoder = MammalEncoder(mammal_model, mammal_tokenizer, device, num_bio_tokens)
except Exception as e:  # pragma: no cover - optional model loading
    llm_model = tokenizer = adapter = trainable_bio = encoder = None

BIO_TOKENS = f"{BIO_START_TOKEN} {' '.join(bio_token_list)} {TRAINABLE_BIO_TOKEN} {BIO_END_TOKEN}"


class PredictRequest(BaseModel):
    cells: List[int]
    question: str


@app.post("/predict")
def predict(req: PredictRequest):
    if any(v is None for v in [llm_model, adapter, tokenizer, trainable_bio, encoder]):
        return {"predictions": []}
    preds = []
    for idx in req.cells:
        expr = (
            adata.X[idx].toarray().flatten()
            if hasattr(adata.X[idx], "toarray")
            else adata.X[idx]
        )
        embedding = encoder.get_cell_embedding(adata.var_names.tolist(), expr)
        prompt = f"{req.question} {BIO_TOKENS} {ANSWER_TOKEN}"
        pred = run_inference(
            llm_model,
            adapter,
            tokenizer,
            trainable_bio,
            prompt,
            embedding,
            device,
        )
        preds.append(pred)
    return {"predictions": preds}


if __name__ == "__main__":  # pragma: no cover - manual start
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
