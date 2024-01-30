# Chess with Reinforcement Learning

This repo attempts to train a RL agent to play chess.

As side effect, this repo also provides examples about how to use [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) for Reinforcement Learning, with and without leverage the new [TorchRL](https://github.com/pytorch/rl) library.

## Pre-requisites

Install Stockfish:

```bash
# MAC
brew install stockfish
```

```bash
# Ubuntu
sudo apt-get install scid
sudo apt-get install stockfish
```

## Installation

Create a Python virtual environment and run:

```bash
pip install --upgrade pip
pip install --upgrade poetry
poetry install
```

## Examples

See [here](./examples).
