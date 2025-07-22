from pathlib import Path

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import scanpy as sc
import numpy as np
from fastapi.middleware.wsgi import WSGIMiddleware

try:
    from langflow.main import setup_app
except Exception as e:  # pragma: no cover - optional langflow
    setup_app = None


def create_dash_app() -> dash.Dash:
    """Return a Dash app showing the single-cell browser."""
    adata = sc.datasets.pbmc68k_reduced()
    if "X_umap" not in adata.obsm:
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
    umap = adata.obsm["X_umap"]
    fig = px.scatter(x=umap[:, 0], y=umap[:, 1], hover_name=adata.obs_names)

    dash_app = dash.Dash(__name__, requests_pathname_prefix="/cells/")
    dash_app.layout = html.Div(
        [
            html.H2("Bioverse scRNA-seq Browser"),
            dcc.Graph(id="cell-plot", figure=fig),
            dcc.Input(id="question", type="text", placeholder="Ask a question"),
            html.Button("Submit", id="submit"),
            html.Div(id="answer"),
        ]
    )

    @dash_app.callback(
        Output("answer", "children"),
        Input("submit", "n_clicks"),
        Input("cell-plot", "clickData"),
        Input("question", "value"),
    )
    def on_submit(n_clicks, click_data, question):  # pragma: no cover - UI logic
        if not n_clicks:
            return ""
        if not click_data:
            return "Please select a cell on the scatter plot."
        cell_idx = click_data["points"][0]["pointIndex"]
        expr = np.asarray(adata[cell_idx].X).flatten().tolist()
        prompt = {"cell_embedding": expr, "question": question or ""}
        return str(prompt)

    return dash_app


def create_combined_app():
    """Create a FastAPI app mounting Langflow and the Dash browser."""
    if setup_app is None:
        raise RuntimeError("Langflow is not installed")
    app = setup_app()
    dash_app = create_dash_app()
    app.mount("/cells", WSGIMiddleware(dash_app.server))
    return app


if __name__ == "__main__":  # pragma: no cover - manual start
    import uvicorn

    uvicorn.run(create_combined_app(), host="0.0.0.0", port=7860)
