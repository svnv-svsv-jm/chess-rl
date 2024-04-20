import pytest
from loguru import logger
import typing as ty
import sys, os

import torch
from torchrl.modules import QValueActor, ValueOperator
from torchrl.objectives import DiscreteCQLLoss

from shark.env import make_chess_env
from shark.models.utils import make_chess_actor_critic


@pytest.mark.parametrize("num_workers", [1, 2])
def test_discretecql_w_chess(engine_executable: str, num_workers: int) -> None:
    """Test website's example + customization."""
    # Model
    env = make_chess_env(engine_executable, num_workers=num_workers)
    actor_nn, value_nn = make_chess_actor_critic(
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
    actor = QValueActor(actor_nn, in_keys=["observation"], action_space=env.action_spec)
    loss_module = DiscreteCQLLoss(actor, action_space=env.action_spec)
    loss = loss_module(td)
    logger.info(f"Loss: {loss}")
    # # Q-Value
    # value_module = ValueOperator(
    #     module=value_nn,
    #     in_keys=["observation", "action"],
    #     out_keys=["state_action_value"],
    # )
    # td = env.reset()
    # td = env.step(env.rand_action(td))
    # td = value_module(td)
    # logger.debug(f"Initialized value_module: {td}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s"])
