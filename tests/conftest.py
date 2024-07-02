import pytest
import os
import sys
from loguru import logger
import typing as ty
import pyrootutils
import torch

from chess.engine import SimpleEngine

from svchess.utils import find_device

from helpers import show_threads, close_all_threads

# Using pyrootutils, we find the root directory of this project and make sure it is our working directory
root = pyrootutils.setup_root(
    search_from=".",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
    cwd=True,
)


@pytest.fixture(scope="session")
def engine_executable() -> str:
    """Chess engine executable path or command."""
    return os.environ.get("CHESS_ENGINE_EXECUTABLE", "stockfish")


@pytest.fixture(scope="session")
def engine(engine_executable: str) -> SimpleEngine:
    """Chess engine."""
    engine = SimpleEngine.popen_uci(engine_executable)
    return engine


@pytest.fixture(scope="session", autouse=True)
def run_before_and_after_tests(engine: SimpleEngine) -> ty.Generator:
    """Fixture to execute code before and after tests are run.
    Use this fixture to set up and tear down stuff.
    """
    # Setup: fill with any logic you want
    logger.info("Setting up...")
    # Return to the tests
    with engine:
        yield  # this is where the testing happens
    # Teardown: fill with any logic you want
    # show_threads()
    # close_all_threads()
    # Finished
    logger.debug("Exiting...")


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Torch device."""
    return find_device("auto")
