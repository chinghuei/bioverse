"""Dash app demonstrating scRNA-seq + LLM integration via Langflow."""

from pathlib import Path

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import scanpy as sc

from langflow import load_flow

import numpy as np



# Load a toy scRNA-seq dataset
adata = sc.datasets.pbmc68k_reduced()

# Ensure we have 2D coordinates for plotting
if "X_umap" not in adata.obsm:
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

umap = adata.obsm["X_umap"]

# Plotly figure for cell selection
fig = px.scatter(x=umap[:, 0], y=umap[:, 1], hover_name=adata.obs_names)

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H2("Bioverse scRNA-seq Demo"),
        dcc.Graph(id="cell-plot", figure=fig),
        dcc.Input(id="question", type="text", placeholder="Ask a question"),
        html.Button("Submit", id="submit"),
        html.Div(id="answer"),
    ]
)


# Langflow workflow used for LLM inference
FLOW_PATH = Path(__file__).parent / "flow.json"
flow = load_flow(FLOW_PATH) if FLOW_PATH.exists() else None


def query_langflow(cell_idx: int, question: str) -> str:
    """Run the Langflow workflow with the given cell and question."""
    expr = np.asarray(adata[cell_idx].X).flatten().tolist()
    # For a full demo you would convert the expression into an embedding using
    # ``MammalEncoder`` or another encoder from ``bioverse``. Here the raw
    # expression vector is passed directly.
    cell_embed = expr

    prompt = {"cell_embedding": cell_embed, "question": question}
    if flow:
        return flow(prompt)
    return "[Langflow workflow missing]"


@app.callback(
    Output("answer", "children"),
    Input("submit", "n_clicks"),
    Input("cell-plot", "clickData"),
    Input("question", "value"),
)
def on_submit(n_clicks, click_data, question):
    if not n_clicks:
        return ""
    if not click_data:
        return "Please select a cell on the scatter plot."
    cell_idx = click_data["points"][0]["pointIndex"]
    answer = query_langflow(cell_idx, question or "")
    return answer


if __name__ == "__main__":
    app.run_server(debug=True)
