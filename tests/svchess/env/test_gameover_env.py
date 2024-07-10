import pytest
from loguru import logger
import typing as ty
import sys, os

import torch
from torch import Tensor
from tensordict import TensorDict

from svchess.env import Chess


@pytest.mark.parametrize("play_as", [True, False])
def test_chess_game(engine_executable: str, device: torch.device, play_as: bool) -> None:
    """Test `Chess` environment manually."""
    # Create env
    env = Chess(
        engine_path=engine_executable,
        device=device,
        play_as=play_as,
    )
    # Play
    while not env.board.is_game_over():
        td = env.sample(from_engine=False)
        td = env.step(td)
        logger.info(td)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
