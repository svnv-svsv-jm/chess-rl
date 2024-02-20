import pytest
from loguru import logger
import typing as ty
import sys

import random
import chess

from shark.utils import remove_illegal_move, uci_to_one_hot_action, move_action_space, action_to_uci


def test_remove_illegal_move() -> None:
    """Test `remove_illegal_move`."""
    # Get all legal moves in UCI format
    board = chess.Board()
    legal_ucis = [m.uci() for m in board.legal_moves]
    # Get all possible moves
    action, mapping = move_action_space()
    # Need to shuffle and retry to make this a valid test
    for _ in range(3):
        # Get the keys and shuffle them
        keys = list(mapping.keys())
        random.shuffle(keys)
        # Find any illegal move
        idx = None
        for uci in keys:
            if uci not in legal_ucis:
                idx = mapping[uci]
                break
        assert isinstance(idx, int)
        # Create illegal action
        action[idx] = 1
        # Remove illegal move
        action = remove_illegal_move(action, board)
        uci = action_to_uci(action)
        # Test move was legalized
        assert board.is_legal(chess.Move.from_uci(uci))
    # Also test this random function just for coverage
    uci_to_one_hot_action(uci)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
