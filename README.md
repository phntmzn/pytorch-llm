# PyTorch LLM

A decoder-only Transformer language model implemented using PyTorch.

## Project Structure

- `config.py` - Model and training configuration
- `train.py` - Training entry point
- `generate.py` - Text generation
- `tokenizer/` - Tokenization
- `model/` - Transformer and GPT architecture
- `training/` - Dataset, training loop, and checkpoints
- `data/` - Training datasets
- `checkpoints/` - Saved model weights
- `tests/` - Unit tests

## Installation

Create a virtual environment:

    python3 -m venv .venv

Activate it on macOS:

    source .venv/bin/activate

Install dependencies:

    python -m pip install -r requirements.txt

## Status

Project structure initialized.
Model implementation is the next step.
