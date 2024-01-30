import pytest
from loguru import logger
import typing as ty
import sys

import chess

from shark.utils import board_to_tensor


def test_board_to_tensor() -> None:
    """Test we can convert a board to a tensor."""
    # Example usage
    chess_board = chess.Board()
    state = board_to_tensor(chess_board)
    logger.info(f"State: {state.size()}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
