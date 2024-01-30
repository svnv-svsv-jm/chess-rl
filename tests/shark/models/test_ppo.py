import pytest
from loguru import logger
import typing as ty
import sys, os

import pandas as pd
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger

from shark.models import PPO


def test_ppo() -> None:
    """Test PPO on InvertedDoublePendulum."""
    frame_skip = 1
    frames_per_batch = frame_skip * 100
    model = PPO(
        env_name="InvertedDoublePendulum-v4",
        frame_skip=frame_skip,
        frames_per_batch=frames_per_batch,
        n_mlp_layers=7,
    )
    # Rollout
    rollout = model.env.rollout(3)
    logger.info(f"Rollout of three steps: {rollout}")
    logger.info(f"Shape of the rollout TensorDict: {rollout.batch_size}")
    logger.info(f"Running policy: {model.policy_module(model.env.reset())}")
    logger.info(f"Running value: {model.value_module(model.env.reset())}")
    # Collector
    collector = model.train_dataloader()
    for _, tensordict_data in enumerate(collector):
        logger.info(f"Tensordict data:\n{tensordict_data}")
        batch_size = int(tensordict_data.batch_size[0])
        assert batch_size == int(frames_per_batch // frame_skip)
        break
    # Training
    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=10,
        val_check_interval=5,
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
