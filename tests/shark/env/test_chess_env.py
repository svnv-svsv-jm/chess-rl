import pytest
from loguru import logger
import typing as ty
import sys, os

from torchrl.envs.utils import check_env_specs

from shark.env import ChessEnv


def test_chess_env(engine_executable: str) -> None:
    """Test we can initialize the chess environment."""
    env = ChessEnv(engine_path=engine_executable, play_as="black")
    # Sanity check
    check_env_specs(env)
    # Reset
    state = env.reset()
    logger.info(f"state: {state}")
    # Step
    action = env.sample()
    out = env.step(action)
    logger.info(f"out: {out}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
