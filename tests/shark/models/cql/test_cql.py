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
    model.advantage(td)
    # Collector
    model.setup()
    collector = model.train_dataloader()
    for _, tensordict_data in enumerate(collector):
        logger.info(f"Tensordict data:\n{tensordict_data}")
        # NOTE: tensordict_data.batch_size has shape [batch_size, rollout_size] or [rollout_size]
        if len(tensordict_data.batch_size) > 1:
            batch_size = int(tensordict_data.batch_size[0])
            assert (
                batch_size == model.num_envs
            ), f"Got batch_size={batch_size} but model.num_envs={model.num_envs}."
            rollout_size = int(tensordict_data.batch_size[1])
            target = int(frames_per_batch // frame_skip)
            assert (
                rollout_size == target
            ), f"Got rollout_size={rollout_size} but int(frames_per_batch // frame_skip)={target}."
        else:
            rollout_size = int(tensordict_data.batch_size[0])
            target = int(frames_per_batch // frame_skip)
            assert (
                rollout_size == target
            ), f"Got rollout_size={rollout_size} but int(frames_per_batch // frame_skip)={target}."
        break
    # Manual step
    for _, batch in enumerate(model.train_dataloader()):
        model.advantage(batch)
        subdata = model.replay_buffer.sample(model.sub_batch_size)
        logger.info(f"Sampled data: {subdata}")
        loss_vals = model.loss(subdata.to(model.device))
        loss, losses = model.collect_loss(loss_vals)
        assert not loss.isnan().any()
        logger.info(losses)
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
