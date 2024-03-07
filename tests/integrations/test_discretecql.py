import pytest
from loguru import logger
import typing as ty
import sys, os

import torch
from tensordict import TensorDict
from torchrl.modules import MLP, QValueActor
from torchrl.data import OneHotDiscreteTensorSpec
from torchrl.objectives import DiscreteCQLLoss

from shark.env import ChessEnv
from shark.models.utils import make_chess_actor_critic


def test_discretecql_w_chess(engine_executable: str) -> None:
    """Test website's example + customization."""
    # Model
    env = ChessEnv(engine_executable)
    actor_nn, _ = make_chess_actor_critic(
        base_env=env,
        critic_type="observation",
        out_features_multiplier=1,
    )
    logger.info(f"Actor: {actor_nn}")
    td = env.reset()
    td = env.rand_action(td)
    td = env.step(td)
    observation: torch.Tensor = td["observation"]
    logger.info(f"Observation: {observation.size()}")
    action: torch.Tensor = actor_nn(observation)
    logger.info(f"Action: {action.size()}")
    # Set up
    actor = QValueActor(actor_nn, in_keys=["observation"], action_space=env.action_spec)
    loss_module = DiscreteCQLLoss(actor, action_space=env.action_spec)
    loss = loss_module(td)
    logger.info(f"Loss: {loss}")


def test_discretecql() -> None:
    """Test website's example."""
    n_obs, n_act = 4, 3
    value_net = MLP(in_features=n_obs, out_features=n_act)
    spec = OneHotDiscreteTensorSpec(n_act)
    actor = QValueActor(value_net, in_keys=["observation"], action_space=spec)
    loss_module = DiscreteCQLLoss(actor, action_space=spec)
    batch = [10]
    data = TensorDict(
        {
            "observation": torch.randn(*batch, n_obs),
            "action": spec.rand(batch),
            ("next", "observation"): torch.randn(*batch, n_obs),
            ("next", "done"): torch.zeros(*batch, 1, dtype=torch.bool),
            ("next", "terminated"): torch.zeros(*batch, 1, dtype=torch.bool),
            ("next", "reward"): torch.randn(*batch, 1),
        },
        batch,
    )
    loss = loss_module(data)
    logger.info(f"Loss: {loss}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
