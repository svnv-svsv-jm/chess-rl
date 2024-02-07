import pytest
from loguru import logger
import typing as ty
import sys, os

import pandas as pd
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from torchrl.envs.utils import check_env_specs
from shark.models import PPOChess
from shark.env import ChessEnv


def test_ppo(engine_executable: str) -> None:
    """Test PPO on InvertedDoublePendulum."""
    model = PPOChess(
        env=ChessEnv(engine_executable),
        frames_per_batch=1,
        total_frames=3,
    )
    # Training
    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=2,
        val_check_interval=1,
        log_every_n_steps=1,
        logger=CSVLogger(
            save_dir="pytest_artifacts",
            name=model.__class__.__name__,
        ),
    )
    trainer.fit(model)
    # Get logged stuff
    log_dir = trainer.log_dir
    assert isinstance(log_dir, str)
    logs = trainer.logged_metrics
    assert isinstance(logs, dict)
    logger.info(log_dir)
    logger.info(logs)
    filename = os.path.join(log_dir, "metrics.csv")
    df = pd.read_csv(filename)
    logger.info(df.head())
    # Plot
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plt.plot(df["reward/train"].to_numpy())
    plt.title("training rewards (average)")
    plt.subplot(2, 2, 2)
    plt.plot(df["step_count/train"].to_numpy())
    plt.title("Max step count (training)")
    plt.subplot(2, 2, 3)
    plt.plot(df["reward_sum/eval"].to_numpy())
    plt.title("Return (test)")
    plt.subplot(2, 2, 4)
    plt.plot(df["step_count/eval"].to_numpy())
    plt.title("Max step count (test)")
    # plt.show()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
