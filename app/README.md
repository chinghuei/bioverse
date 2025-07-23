# Bioverse Demo App

This demo uses [Dash](https://dash.plotly.com/) to visualise a scRNA-seq dataset
and [Langflow](https://github.com/logspace-ai/langflow) for LLM integration.
Users can select one or more cells using the lasso tool on the scatter plot,
provide a question and obtain cell type predictions via the Langflow workflow
or the built in prediction server.

The application reads a `flow.json` file from this directory. If the JSON
contains a `data` field the corresponding Langflow workflow is loaded. When the
file only includes a `config` section with an `endpoint`, the app skips loading
the workflow and forwards requests to that MCP endpoint (default
`http://localhost:8000/predict`).

## Running

Install the required packages (base dependencies come from the project
root):

```bash
pip install -r ../requirements.txt
pip install -r requirements.txt
```

To launch the stand-alone Dash demo run:

```bash
python app.py
```

The inference server can be started with:

```bash
python mcp_server.py
```

### Running inside Langflow

To start Langflow with the single-cell browser mounted under `/cells` run:

```bash
python langflow_server.py
```

A browser window will open showing the UMAP projection of the example dataset.
Select one or more cells using the lasso tool, enter a question and press
**Submit** to send the request through the Langflow workflow and display the
predicted cell type(s).

