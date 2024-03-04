import pytest
from loguru import logger
import typing as ty
import sys, os

import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from tensordict import TensorDict
from torchrl.objectives import CQLLoss

from shark.callbacks import DebugCallback
from shark.models import CQLChess
from shark.utils import get_logged_metrics_from_trainer, plot_metrics


@pytest.mark.parametrize("automatic_optimization", [False, True])
def test_cql(engine_executable: str, automatic_optimization: bool) -> None:
    """Test PPO on InvertedDoublePendulum."""
    model = CQLChess(
        engine_executable=engine_executable,
        depth=1,
        n_mlp_layers=1,
        num_mlp_cells=32,
        num_cells=32,
        frames_per_batch=2,
        total_frames=10,
        automatic_optimization=automatic_optimization,
        env_kwargs=dict(lose_on_illegal_move=False),
        num_envs=1,
    )
    assert isinstance(model.loss_module, CQLLoss)
    # Try to manually run training loop
    # So we can decouple implementation from Lightning errors
    loader = model.train_dataloader()
    cfg = model.configure_optimizers()
    optimizer = cfg["optimizer"]
    for batch in loader:
        model.advantage(batch)
        subdata: TensorDict = model.replay_buffer.sample(model.sub_batch_size)
        loss_vals = model.loss(subdata.to(model.device))
        loss, losses = model.collect_loss(loss_vals)
        logger.info(losses)
        optimizer.zero_grad()  # type: ignore
        loss.backward()
        logger.success(f"Able to run optim step.")
        break
    # Training
    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=2,
        val_check_interval=1,
        log_every_n_steps=1,
        callbacks=DebugCallback(level="DEBUG"),
        logger=CSVLogger(
            save_dir="pytest_artifacts",
            name=model.__class__.__name__,
        ),
    )
    trainer.fit(model)
    # Get logged stuff
    df: pd.DataFrame = get_logged_metrics_from_trainer(trainer)
    # Plot
    plot_metrics(df)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
