import pytest
from loguru import logger
import typing as ty
import sys, os

import pandas as pd
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from tensordict import TensorDict

from shark.callbacks import DebugCallback
from shark.models import PPOChess
from shark.utils import get_logged_metrics_from_trainer


@pytest.mark.parametrize("automatic_optimization", [False, True])
def test_ppo(engine_executable: str, automatic_optimization: bool) -> None:
    """Test PPO on InvertedDoublePendulum."""
    model = PPOChess(
        engine_executable=engine_executable,
        depth=1,
        n_mlp_layers=1,
        num_mlp_cells=32,
        num_cells=32,
        frames_per_batch=2,
        total_frames=10,
        automatic_optimization=automatic_optimization,
        use_one_hot=False,
        chess_env_kwargs=dict(lose_on_illegal_move=False),
    )
    # Try to manually run training loop
    # So we can decouple implementation from Lightning errors
    loader = model.train_dataloader()
    cfg = model.configure_optimizers()
    optimizer = cfg["optimizer"]
    for batch in loader:
        model.advantage_module(batch)
        subdata: TensorDict = model.replay_buffer.sample(model.sub_batch_size)
        loss_vals = model.loss_module(subdata.to(model.device))
        loss, losses = model.loss(loss_vals)
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
        callbacks=DebugCallback(level="TRACE"),
        logger=CSVLogger(
            save_dir="pytest_artifacts",
            name=model.__class__.__name__,
        ),
    )
    trainer.fit(model)
    # Get logged stuff
    df: pd.DataFrame = get_logged_metrics_from_trainer(trainer)
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
    plt.savefig(os.path.join("pytest_artifacts", "metrics.png"))
    # plt.show()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
