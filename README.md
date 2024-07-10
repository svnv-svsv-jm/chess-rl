# Chess with Reinforcement Learning

[![pipeline status](https://gitlab.com/gianmarcoaversanotest/chess/badges/main/pipeline.svg)](https://gitlab.com/gianmarcoaversanotest/chess/-/commits/main) [![coverage report](https://gitlab.com/gianmarcoaversanotest/chess/badges/main/coverage.svg)](https://gitlab.com/gianmarcoaversanotest/chess/-/commits/main) [![Latest Release](https://gitlab.com/gianmarcoaversanotest/chess/-/badges/release.svg)](https://gitlab.com/gianmarcoaversanotest/chess/-/releases)

This repository implements a Chess RL environment using [TorchRL](https://github.com/pytorch/rl).

## Idea

The core idea is to solve the sparse reward problem of RL agents by using a pretrained chess engine (e.g. Stockfish) to provide feedback (reward signal) for each move the agent makes. Besides, the pretrained chess engine can also be used as opponent player.

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

Create a Python virtual environment and, from the project's root folder, run:

```bash
pip install --upgrade pip
pip install --upgrade poetry
poetry self update
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
