# Bioverse

This repository contains example scripts and utilities for training and
inference with Bioverse models. The code relies on PyTorch, Scanpy and
Transformers as well as the "MAMMAL" encoder for gene expression data.

The main entry points are:

* `train.py` – fine‑tune the LLM on gene expression embeddings.
* `evaluate.py` – evaluate a saved model on a held‑out test set.
* `infer.py` – run inference on random samples of the dataset.
* `app/` – small Dash demo and FastAPI inference server.

The instructions below assume a Unix‑like environment with **Python 3.9–3.11**
installed.

## Installation

1. Create a virtual environment with your preferred Python (>=3.9) and activate
   it:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install the dependencies listed in `requirements.txt` (this may take a
   while as PyTorch and Scanpy download sizeable wheels):

   ```bash
   pip install -r requirements.txt
   ```

   The demo application under `app/` has a separate requirements file that
   should be installed in addition to the base dependencies when running the
   web UI.

## Training a model

1. Edit `train.py` if you want to change hyper‑parameters such as learning
   rate or number of epochs.
2. Download or provide an AnnData `.h5ad` file compatible with the
   `load_AnnData_from_file` function in `bioverse.data`.
3. Start training (optionally specify the dataset path):

   ```bash
   python train.py --data path/to/your_dataset.h5ad
   ```

   Checkpoints are written to the `checkpoints/` directory. After training the
   final model is stored in `checkpoints/final_model/`.

## Evaluating a model

Once a model has been trained you can compute metrics on a test set:

```bash
python evaluate.py --data path/to/your_dataset.h5ad
```

Results are printed to the console and logged to ClearML if it is
configured.

## Running ad‑hoc inference

```bash
python infer.py --data path/to/your_dataset.h5ad
```

This loads the saved model and prints predictions for a random subset of
cells.

## Demo application

Inside `app/` you can run a small web demo that visualises a toy scRNA‑seq
UMAP plot and queries the model through a Langflow workflow or the built‑in
FastAPI server.

```bash
cd app
pip install -r requirements.txt
python app.py            # launches the Dash front end on port 8051
python mcp_server.py     # optional: start the FastAPI server on port 8000
```

Open <http://localhost:8051> in your browser to interact with the demo.

