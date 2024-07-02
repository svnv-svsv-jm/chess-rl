import pytest
from loguru import logger
import typing as ty
import sys, os

from tensordict import TensorDict
from torchrl.envs import check_env_specs

from svchess.env import Chess
from svchess.utils import find_device


@pytest.mark.parametrize("use_one_hot", [True, False])
def test_chess_env(engine_executable: str, use_one_hot: bool) -> None:
    """Test we can initialize the chess environment."""
    # Create env
    env = Chess(
        # engine_path=engine_executable,
        # play_as="black",
        # device=find_device(),
        # use_one_hot=use_one_hot,
        # probability_move_is_random=0.5,
    )
    # # Reset
    # state = env.reset()
    # logger.info(f"Reset state: {state}")
    # # Step
    # action = env.sample()
    # logger.info(f"Action: {action}")
    # td: TensorDict = env.step(action)
    # logger.info(f"Tensordict: {td}")
    # # Rollout
    # td = env.rollout(3)
    # logger.info(f"Rollout: {td}")
    # Sanity check
    check_env_specs(env)
    # # Sample
    # env.sample(from_engine=False)
    # env.sample(from_engine=True)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
