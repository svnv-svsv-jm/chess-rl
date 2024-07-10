import pytest
import os
import sys
from loguru import logger
import typing as ty
import pyrootutils
import torch

import chess
from chess.engine import SimpleEngine

from svchess.utils import find_device

# Using pyrootutils, we find the root directory of this project and make sure it is our working directory
root = pyrootutils.setup_root(
    search_from=".",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
    cwd=True,
)


@pytest.fixture
def engine_executable() -> str:
    """Chess engine executable path or command."""
    return os.environ.get("CHESS_ENGINE_EXECUTABLE", "stockfish")


@pytest.fixture
def engine(engine_executable: str) -> SimpleEngine:  # type: ignore
    """Chess engine."""
    with SimpleEngine.popen_uci(engine_executable) as engine_:
        yield engine_


@pytest.fixture
def device() -> torch.device:
    """Torch device."""
    return find_device("auto")
