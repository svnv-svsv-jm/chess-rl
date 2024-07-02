import pytest
from loguru import logger
import typing as ty
import sys

import chess

from svchess.utils import board_to_tensor


@pytest.mark.parametrize(
    "flatten, one_hot",
    [
        (True, False),
        (False, True),
        (False, False),
        (True, True),
    ],
)
def test_board_to_tensor(flatten: bool, one_hot: bool) -> None:
    """Test `board_to_tensor()`."""
    board = board_to_tensor(chess.Board(), flatten=flatten, one_hot=one_hot)
    logger.info(f"Board: {board.size()}")
    # Tests
    if flatten:
        assert board.dim() == 1
        if one_hot:
            assert board.numel() == 832
        else:
            assert board.numel() == 64
    else:
        assert board.dim() > 1
        if one_hot:
            assert board.numel() == 832
            assert board.size() == (8, 8, 13)
        else:
            assert board.numel() == 64
            assert board.size() == (8, 8)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
