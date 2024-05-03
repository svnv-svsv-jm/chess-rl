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
from shark.models.utils import initialize_actor, make_chess_actor_critic


@pytest.mark.parametrize(
    "builtin, random_policy",
    [
        (True, True),
        (False, True),
        (True, False),
        (False, False),
    ],
)
def test_collector(engine_executable: str, builtin: bool, random_policy: bool) -> None:
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
    if random_policy or not isinstance(env, ChessEnv):
        policy = RandomPolicy(env.action_spec)
    else:
        actor_nn, _ = make_chess_actor_critic(env)
        policy = initialize_actor(
            actor_nn=actor_nn,
            env=env,
            flatten_state=False,
            qvalue=False,
        )
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
