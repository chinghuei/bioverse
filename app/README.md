# Bioverse Demo App

This demo uses [Dash](https://dash.plotly.com/) to visualise a scRNA-seq dataset
and [Langflow](https://github.com/logspace-ai/langflow) for LLM integration.
Users can select a single cell from a scatter plot, provide a question and obtain
an answer generated via a Langflow workflow.

## Running

Install the required packages:

```bash
pip install -r requirements.txt
```

Then start the app:

```bash
python app.py
```

A browser window will open showing the UMAP projection of the example dataset.
Select a cell, enter a question and press **Submit** to send the request through
the Langflow workflow and display the response.
