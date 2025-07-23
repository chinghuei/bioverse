"""Dash app demonstrating scRNA‑seq + LLM integration via Langflow."""

from pathlib import Path
import json
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import requests
from typing import List
import plotly.express as px
import scanpy as sc

# ✅ Updated import based on current Langflow structure
from langflow.load import load_flow_from_json

# --- Load toy scRNA‑seq dataset
adata = sc.datasets.pbmc68k_reduced()
if "X_umap" not in adata.obsm:
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
umap = adata.obsm["X_umap"]

# --- Plotly figure
fig = px.scatter(x=umap[:, 0], y=umap[:, 1], hover_name=adata.obs_names)
fig.update_layout(dragmode="lasso")

# --- Dash app setup
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H2("Bioverse scRNA‑seq + Langflow Demo"),
    dcc.Graph(id="cell-plot", figure=fig),
    dcc.Input(id="question", type="text", placeholder="Ask a question"),
    html.Button("Submit", id="submit"),
    html.Div(id="answer"),
])

# --- Load Langflow workflow
FLOW_PATH = Path(__file__).parent / "flow.json"
flow = None
PREDICT_URL = "http://localhost:8000/predict"
if FLOW_PATH.exists():
    try:
        flow_json = json.loads(FLOW_PATH.read_text())
        if "config" in flow_json and "endpoint" in flow_json["config"]:
            PREDICT_URL = flow_json["config"]["endpoint"]
        if "data" in flow_json:
            flow = load_flow_from_json(str(FLOW_PATH), build=False)
    except Exception:
        pass

# --- Langflow query handler


def query_langflow(cell_indices: List[int], question: str) -> str:
    payload = {"cells": cell_indices, "question": question}
    if flow:
        try:
            return flow(payload)
        except Exception:
            pass
    try:
        resp = requests.post(PREDICT_URL, json=payload, timeout=30)
        if resp.status_code == 200:
            result = resp.json().get("predictions", [])
            return "; ".join(result)
        else:
            try:
                error_msg = resp.json().get("error")
            except ValueError:
                error_msg = resp.text
            return f"[{error_msg}]"
    except Exception as exc:
        return f"[Prediction error: {exc}]"
    return "[Langflow workflow missing]"

# --- Callback for question handling
@app.callback(
    Output("answer", "children"),
    Input("submit", "n_clicks"),
    Input("cell-plot", "selectedData"),
    Input("question", "value"),
)
def on_submit(n_clicks, selected_data, question):
    if not n_clicks:
        return ""
    if not selected_data or not selected_data.get("points"):
        return "Please select at least one cell on the scatter plot."
    indices = [p["pointIndex"] for p in selected_data["points"]]
    return query_langflow(indices, question or "")

# --- Run the Dash server (✅ updated method)
if __name__ == "__main__":
    app.run(debug=True, port=8051)
