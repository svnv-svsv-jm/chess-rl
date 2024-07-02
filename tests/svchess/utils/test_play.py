import pytest
from loguru import logger
import typing as ty
import sys

import chess
from chess.engine import SimpleEngine

from svchess.utils import play_move


def test_play_move_by_starting_engine(engine_executable: str) -> None:
    """Test `play_move()`."""
    board = chess.Board()
    board = play_move(board, engine_executable=engine_executable)


def test_play_move_from_engine(engine: SimpleEngine) -> None:
    """Test `play_move()`."""
    board = chess.Board()
    board = play_move(board, engine=engine)


def test_play_move_random() -> None:
    """Test `play_move()`."""
    board = chess.Board()
    board = play_move(board)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
