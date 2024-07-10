import pytest
from loguru import logger
import typing as ty
import sys, os

import torch
from torch import Tensor
from tensordict import TensorDict
from torchrl.envs import check_env_specs

from svchess.env import Chess
from svchess.utils import find_device
from svchess.utils.const import INITIAL_STATE


@pytest.mark.parametrize("play_as", [True, False])
def test_chess_reset(engine_executable: str, device: torch.device, play_as: bool) -> None:
    """Test `Chess` environment manually."""
    # Create env
    env = Chess(
        engine_path=engine_executable,
        device=device,
        play_as=play_as,
    )
    # Reset
    td: TensorDict = env.reset()
    state: Tensor = td["state"]
    logger.info(f"Reset state: {state.size()}")
    logger.info(f"INITIAL_STATE: {INITIAL_STATE.size()}")
    # Tests
    eq = state.cpu() == INITIAL_STATE
    if play_as:
        # If playing as white, we expect the initial state
        assert eq.all()
    else:
        # If playing as black, we expect all equal except 2 positions
        assert eq.sum() == eq.numel() - 2


def test_chess_env(engine_executable: str) -> None:
    """Test `Chess` environment."""
    # Create env
    env = Chess(engine_path=engine_executable, device=find_device("auto"))
    # Sanity check
    check_env_specs(env)


def test_chess_env_all_inputs(device: torch.device) -> None:
    """Test `Chess` environment: try to cover all initializations."""
    Chess(device=device)
    Chess(engine_path="yo", device=device)


def test_chess_env_manually(engine_executable: str, device: torch.device) -> None:
    """Test `Chess` environment manually."""
    # Create env
    env = Chess(engine_path=engine_executable, device=device)
    # Reset
    state = env.reset()
    logger.info(f"Reset state: {state}")
    # Step
    action = env.sample(from_engine=False)
    action = env.sample(from_engine=True)
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
