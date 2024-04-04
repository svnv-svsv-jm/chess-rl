import pytest
from loguru import logger
import typing as ty
import sys

import lightning.pytorch as pl

from shark.models import DQN


def test_dqn_on_cartpole() -> None:
    """Test DQN on CartPole-v0."""
    model = DQN(env="CartPole-v0")
    trainer = pl.Trainer(accelerator="cpu", max_steps=300)
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with status {trainer.state}"


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
