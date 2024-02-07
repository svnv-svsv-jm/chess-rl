import pytest
from loguru import logger
import typing as ty
import sys

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from shark.utils import find_device
from shark.nn import MLP
from shark.env import ChessEnv
from shark.datasets import CollectorDataset


def test_chess_env(engine_executable: str) -> None:
    """Test we can initialize the chess environment."""
    env = ChessEnv(engine_path=engine_executable, play_as="black")
    out_shape = env.action_spec.shape[-1]
    actor_net = MLP(
        out_features=2 * out_shape,
        hidden_dims=[16] * 2,
        tanh=True,
        last_activation=NormalParamExtractor(),
        flatten=True,
        flatten_start_dim=0,
    )
    policy_module = TensorDictModule(
        actor_net,
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )
    td = env.reset()
    policy_module(td)
    collector = CollectorDataset(
        env,
        policy_module,
        frames_per_batch=10,
        total_frames=20,
        device=find_device("cpu"),
    )
    for _, td in enumerate(collector):
        assert isinstance(td, TensorDict)
        break


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
