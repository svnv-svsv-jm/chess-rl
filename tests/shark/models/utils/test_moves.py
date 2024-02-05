import pytest
from loguru import logger
import typing as ty
import sys

import chess

from shark.utils import action_to_one_hot_legal, move_action_space, get_move_score


def test_get_move_score(engine_executable: str) -> None:
    """Test we can score a move."""
    board = chess.Board()
    legal_moves = list(board.legal_moves)
    logger.info(f"Legal moves: {legal_moves}")
    for move in legal_moves:
        score = get_move_score(board, move, engine_path=engine_executable, depth=20)
        logger.info(f"Score: {score}")


def test_move_action_space() -> None:
    """Test `move_action_space()`."""
    # Example usage
    action_space, move_to_index = move_action_space()
    # Print the one-hot matrix (sparse representation for clarity)
    logger.info(f"action_space: {action_space.size()}")
    for key, val in move_to_index.items():
        logger.info(f"{key}: {val}")
        break


def test_action_to_one_hot_legal() -> None:
    """Test `action_to_one_hot_legal()`."""
    # Example usage
    chess_board = chess.Board()
    # Suppose we choose a move
    selected_move = chess.Move.from_uci("e2e4")
    # Convert the selected move to one-hot encoding
    action_one_hot = action_to_one_hot_legal(selected_move.uci(), chess_board)
    logger.info(f"action_one_hot: {action_one_hot}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
