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
    env = Chess()
    # Sanity check
    check_env_specs(env)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
