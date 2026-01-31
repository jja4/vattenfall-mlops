# Vattenfall

ML Service for predicting Finland's electricity imbalance prices.

## Structure
- `app/`: FastAPI application.
- `ingestion/`: Tools for fetching and storing Fingrid data.
- `models/`: Training scripts and model artifacts.
- `data/`: Local data storage (ignored by git).

## Setup
1. Install `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Sync dependencies: `uv sync`
3. Set `.env` keys.
