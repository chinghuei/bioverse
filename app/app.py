"""Dash app demonstrating scRNA‑seq + LLM integration via Langflow."""

from pathlib import Path
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import scanpy as sc
import numpy as np

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
if FLOW_PATH.exists():
    flow = load_flow_from_json(str(FLOW_PATH), build=False)

# --- Langflow query handler
def query_langflow(cell_idx: int, question: str) -> str:
    expr = np.asarray(adata[cell_idx].X).flatten().tolist()
    prompt = {"cell_embedding": expr, "question": question}
    if flow:
        return flow(prompt)
    return "[Langflow workflow missing]"

# --- Callback for question handling
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
    idx = click_data["points"][0]["pointIndex"]
    return query_langflow(idx, question or "")

# --- Run the Dash server (✅ updated method)
if __name__ == "__main__":
    app.run(debug=True, port=8051)