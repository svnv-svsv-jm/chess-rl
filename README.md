# Chess with Reinforcement Learning

[![pipeline status](https://gitlab.com/gianmarcoaversanotest/chess/badges/main/pipeline.svg)](https://gitlab.com/gianmarcoaversanotest/chess/-/commits/main) [![coverage report](https://gitlab.com/gianmarcoaversanotest/chess/badges/main/coverage.svg)](https://gitlab.com/gianmarcoaversanotest/chess/-/commits/main) [![Latest Release](https://gitlab.com/gianmarcoaversanotest/chess/-/badges/release.svg)](https://gitlab.com/gianmarcoaversanotest/chess/-/releases)

This repo attempts to train a RL agent to play chess.

As side effect, this repo also provides examples about how to use [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) for Reinforcement Learning, with the new [TorchRL](https://github.com/pytorch/rl) library.

## Pre-requisites

Install Stockfish:

```bash
# MAC
brew install stockfish
```

```bash
# Ubuntu
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

## Experiments

This project uses Hydra to configure experiments.

You can train a new model by configuring the [configuration file](./configs/main.yaml), then running

```bash
python experiments/main.py # + any Hydra overrides
```

Alternatively, you can train a model as follows.

### Supervisor

Using `supervisor`:

```bash
supervisord -c supervisord.conf
# or
make run
```

### Docker

Using `docker`:

```bash
make exp
# or if you have NVIDIA
exp-gpu
```

## TODO

- `resume-from-checkpoint` option.
- CQL
- pre-trained models
