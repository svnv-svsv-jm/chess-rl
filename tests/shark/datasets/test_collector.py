import pytest
from loguru import logger
import typing as ty
import sys

from tensordict import TensorDict
from torchrl.collectors import RandomPolicy
from torchrl.envs import GymEnv

from shark.env import ChessEnv
from shark.datasets import CollectorDataset
from shark.utils import find_device


@pytest.mark.parametrize("builtin", [True, False])
def test_collector(engine_executable: str, builtin: bool) -> None:
    """Test `CollectorDataset` on built-in gym envs."""
    device = find_device()
    env = (
        GymEnv("CartPole-v1", device=device)
        if builtin
        else ChessEnv(
            engine_executable,
            device=device,
            lose_on_illegal_move=False,
        )
    )
    env.set_seed(0)
    policy = RandomPolicy(env.action_spec)
    collector = CollectorDataset(
        env,
        policy,
        frames_per_batch=2,
        total_frames=10,
        device=device,
    )
    for i, td in enumerate(collector):
        assert isinstance(td, TensorDict)
        if i > 2:
            break
    td = collector.sample()
    logger.info(f"Sample:\n{td}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
