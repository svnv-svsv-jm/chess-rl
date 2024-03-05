import pytest
from loguru import logger
import typing as ty
import sys, os

import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger

from shark.models import CQLPendulum
from shark.utils import get_logged_metrics_from_trainer


def test_cql() -> None:
    """Test CQL on InvertedDoublePendulum."""
    frame_skip = 1
    frames_per_batch = frame_skip * 5
    total_frames = 100
    model = CQLPendulum(
        frame_skip=frame_skip,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        n_mlp_layers=4,
        use_checkpoint_callback=True,
    )
    # Rollout
    env = model.env
    rollout = env.rollout(3)
    logger.info(f"Rollout of three steps: {rollout}")
    logger.info(f"Shape of the rollout TensorDict: {rollout.batch_size}")
    logger.info(f"Env reset: {env.reset()}")
    logger.info(f"Running policy: {model.policy_module(env.reset())}")
    td = env.reset()
    td = env.rand_action(td)
    td = env.step(td)
    logger.info(f"Running value: {model.value_module(td)}")
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
    max_steps = 4
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
    df: pd.DataFrame = get_logged_metrics_from_trainer(trainer)
    logger.info(df.head())
    # Test eval loop was run and returned metrics
    assert "reward/eval" in df.columns


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
