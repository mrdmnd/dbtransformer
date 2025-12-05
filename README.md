# dbtransformer

Train a relational transformer on any database!

This is intended to be run both locally (for testing) and on Prime Intellect GPU servers.

## Datasets

- [RelBench](https://huggingface.co/datasets/relbench/relbench)
- [CTU Prague]
- ???

Data is stored locally in the `data/` directory for now - we'll want to move this to shared storage at some point.

## Setup

```bash
pre-commit clean
pre-commit install
pre-commit install-hooks
uv sync
```

## Run

```bash
uv run main.py
```
