import pytest
from loguru import logger
import typing as ty
import sys, os

import uuid, tempfile
import torch
from torchrl.modules import QValueActor
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.objectives import DiscreteCQLLoss, SoftUpdate
from torchrl.trainers import LogReward, Recorder, ReplayBufferTrainer, Trainer, UpdateWeights
from torchrl.record.loggers.csv import CSVLogger
from torchrl.envs import ExplorationType

from shark.env import make_chess_env
from shark.models.utils import make_chess_actor_critic
from shark.datasets import make_chess_collector


@pytest.mark.parametrize(
    "parallel, num_workers, num_collectors, collector_type",
    [
        (False, 1, 1, "sync"),  # PASSED
        (False, 1, 2, "multi-sync"),  # PASSED
        (False, 1, 2, "multi-async"),  # PASSED
        # (False, 1, 1, "multi-sync"),  # FAILED: object of type 'TransformedEnv' has no len()
        # (True, 1, 1, "sync"),  # FAILED: trying to close a closed environment
        # (True, 2, 2, "sync"),  # FAILED: list object is not callable
        # (True, 1, 2, "sync"),  # FAILED: list object is not callable
        # (True, 1, 2, "multi-sync"),  # FAILED: trying to close a closed environment
        # (True, 1, 2, "multi-async"),  # FAILED: trying to close a closed environment
    ],
)
def test_discretecql_w_chess(
    engine_executable: str,
    parallel: bool,
    num_workers: int,
    num_collectors: int,
    collector_type: str,
) -> None:
    """Test website's example + customization."""
    # Model
    env = make_chess_env(
        engine_executable=engine_executable,
        num_workers=num_workers,
        parallel=parallel,
    )
    actor_nn, _ = make_chess_actor_critic(
        base_env=env,
        critic_type="observation",
        out_features_multiplier=1,
    )
    logger.info(f"Actor: {actor_nn}")
    # Info about env
    td = env.reset()
    td = env.rand_action(td)
    td = env.step(td)
    observation: torch.Tensor = td["observation"]
    logger.info(f"Observation: {observation.size()}")
    action: torch.Tensor = actor_nn(observation)
    logger.info(f"Action: {action.size()}")
    # Set up
    policy_module = QValueActor(actor_nn, in_keys=["observation"], action_space=env.action_spec)
    loss_module = DiscreteCQLLoss(policy_module, action_space=env.action_spec)
    loss = loss_module(td)
    logger.info(f"Loss: {loss}")
    # Set up for training
    exp_name = f"pytest_{uuid.uuid1()}"
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_logger = CSVLogger(exp_name=exp_name, log_dir=tmpdir)
        optimizer = torch.optim.Adam(loss_module.parameters(), lr=1e-3)
        # Create env and data collector
        device = torch.device("cpu")
        collector = make_chess_collector(
            engine_executable=engine_executable,
            actor=policy_module,
            num_collectors=num_collectors,
            device=device,
            collector_type=collector_type,
            num_workers=num_workers,
            parallel=parallel,
        )
        try:
            # Set up trainer
            trainer = Trainer(
                collector=collector,
                total_frames=10,
                frame_skip=1,
                loss_module=loss_module,
                optimizer=optimizer,
                logger=csv_logger,
                optim_steps_per_batch=1,
                log_interval=1,
            )
            # Register hooks
            buffer_hook = ReplayBufferTrainer(
                TensorDictReplayBuffer(batch_size=4, storage=LazyMemmapStorage(100), prefetch=5),
                flatten_tensordicts=True,
            )
            buffer_hook.register(trainer)
            weight_updater = UpdateWeights(collector, update_weights_interval=1)
            weight_updater.register(trainer)
            recorder = Recorder(
                record_interval=5,  # log every 100 optimization steps
                record_frames=50,  # maximum number of frames in the record
                frame_skip=1,
                policy_exploration=policy_module,
                environment=env,
                exploration_type=ExplorationType.MODE,
                log_keys=[("next", "reward")],
                out_keys={("next", "reward"): "rewards"},
                log_pbar=True,
            )
            recorder.register(trainer)
            target_updater = SoftUpdate(loss_module, eps=0.995)
            trainer.register_op("post_optim", target_updater.step)
            log_reward = LogReward(log_pbar=True)
            log_reward.register(trainer)
            # Train
            trainer.train()
        finally:
            # Teardown
            collector.shutdown()
            del collector


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
