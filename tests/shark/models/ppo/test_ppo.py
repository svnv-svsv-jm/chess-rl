import pytest
from loguru import logger
import typing as ty
import sys, os

import pandas as pd
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger

from shark.models import PPOPendulum
from shark.utils import plot_metrics


def test_ppo() -> None:
    """Test PPO on InvertedDoublePendulum."""
    frame_skip = 1
    frames_per_batch = frame_skip * 5
    total_frames = 100
    model = PPOPendulum(
        frame_skip=frame_skip,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        n_mlp_layers=4,
    )
    # Rollout
    rollout = model.env.rollout(3)
    logger.info(f"Rollout of three steps: {rollout}")
    logger.info(f"Shape of the rollout TensorDict: {rollout.batch_size}")
    logger.info(f"Env reset: {model.env.reset()}")
    logger.info(f"Running policy: {model.policy_module(model.env.reset())}")
    logger.info(f"Running value: {model.value_module(model.env.reset())}")
    # Collector
    model.setup()
    collector = model.train_dataloader()
    for _, tensordict_data in enumerate(collector):
        logger.info(f"Tensordict data:\n{tensordict_data}")
        batch_size = int(tensordict_data.batch_size[0])
        rollout_size = int(tensordict_data.batch_size[1])
        assert rollout_size == int(frames_per_batch // frame_skip)
        assert batch_size == model.num_envs
        break
    # Training
    max_steps = 2
    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=max_steps,
        val_check_interval=2,
        log_every_n_steps=1,
        logger=CSVLogger(
            save_dir="pytest_artifacts",
            name=model.__class__.__name__,
        ),
    )
    trainer.fit(model)
    assert max_steps >= trainer.global_step
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
    plot_metrics(df)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
