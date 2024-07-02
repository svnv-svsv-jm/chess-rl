import pytest
from loguru import logger
import typing as ty
import sys, os

import torch
from tensordict import TensorDict
from torchrl.envs import check_env_specs

from svchess.env import Chess
from svchess.utils import find_device


def test_chess_env(engine_executable: str) -> None:
    """Test `Chess` environment."""
    # Create env
    env = Chess(engine_path=engine_executable, device=find_device("auto"))
    # Sanity check
    check_env_specs(env)


def test_chess_env_manually(engine_executable: str) -> None:
    """Test `Chess` environment."""
    device = find_device("auto")
    # Create env
    Chess(device=device)
    Chess(engine_path="yo", device=device)
    env = Chess(engine_path=engine_executable, device=device)
    # Reset
    state = env.reset()
    logger.info(f"Reset state: {state}")
    # Step
    action = env.sample(False)
    action = env.sample(True)
    logger.info(f"Action: {action}")
    td = env.step(action)
    logger.info(f"Tensordict: {td}")
    # Reset
    state = env.reset(td)
    logger.info(f"Reset state: {state}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
