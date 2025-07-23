from pathlib import Path
"""Minimal FastAPI inference server for Bioverse models.

The server exposes a single ``/predict`` endpoint which takes a list of cell
indices and a question string. It loads a toy PBMC dataset and all model
artifacts once at startup so that incoming requests only need to run the
forward pass. This keeps latency low and avoids repeated initialisation.
"""

from pathlib import Path
from typing import List

import torch
import scanpy as sc
from fastapi import FastAPI
from fastapi.responses import JSONResponse
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

# Create the FastAPI application
app = FastAPI()

# ---------------------------------------------------------------------------
# Data and model initialisation
# ---------------------------------------------------------------------------
# The example PBMC dataset is loaded only once when the server starts.
# This keeps the predict endpoint lightweight and avoids disk I/O on each call.
adata = sc.datasets.pbmc68k_reduced()
if "X_umap" not in adata.obsm:
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

# Use GPU if available; otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model components from the checkpoints folder. If files are
# missing we simply disable prediction instead of failing at startup.
model_available = True
try:
    llm_model, tokenizer, adapter, trainable_bio = loadSavedModels(
        Path("checkpoints/final_model"), device
    )
    mammal_model, mammal_tokenizer = loadMammal(
        "ibm/biomed.omics.bl.sm.ma-ted-458m", device
    )
    encoder = MammalEncoder(mammal_model, mammal_tokenizer, device, num_bio_tokens)
except Exception as e:  # pragma: no cover - optional model loading
    print(f"Model loading failed: {e}")
    llm_model = tokenizer = adapter = trainable_bio = encoder = None
    model_available = False

BIO_TOKENS = f"{BIO_START_TOKEN} {' '.join(bio_token_list)} {TRAINABLE_BIO_TOKEN} {BIO_END_TOKEN}"


class PredictRequest(BaseModel):
    cells: List[int]
    question: str


@app.post("/predict")
def predict(req: PredictRequest):
    """Return model predictions for the selected cell indices."""
    # If model components failed to load return an informative error
    if not model_available or any(
        v is None for v in [llm_model, adapter, tokenizer, trainable_bio, encoder]
    ):
        return JSONResponse({"error": "Model not loaded"}, status_code=503)

    preds = []
    for idx in req.cells:
        # Convert the expression profile of the requested cell into a fixed
        # size embedding using the MAMMAL encoder.
        expr = (
            adata.X[idx].toarray().flatten()
            if hasattr(adata.X[idx], "toarray")
            else adata.X[idx]
        )
        embedding = encoder.get_cell_embedding(adata.var_names.tolist(), expr)

        # Build the full prompt and run inference through the language model.
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
