import pytest
from loguru import logger
import typing as ty
import sys

import chess
from chess.engine import SimpleEngine

from svchess.utils import engine_eval

# Create a board with a mate in one board for white
MATE_IN_ONE_FOR_WHITE = chess.Board("5Q2/5K1k/8/8/8/8/8/8 w - - 0 1")


def _tests(r: float, is_white: bool, expected: float) -> None:
    """Tests."""
    play_as = "white" if is_white else "black"
    logger.info(f"Reward ({play_as}): {r}")
    # Tests
    assert r == pytest.approx(expected, abs=6)


@pytest.mark.parametrize(
    "board, is_white, worst_reward, expected_reward",
    [
        (MATE_IN_ONE_FOR_WHITE, False, -1000, -1000),
        (MATE_IN_ONE_FOR_WHITE, True, -100, 100),
        (chess.Board(), False, -100, -50),
        (chess.Board(), True, -100, 50),
        (MATE_IN_ONE_FOR_WHITE.mirror(), True, -100, -100),
    ],
)
def test_engine_eval_from_path(
    engine_executable: str,
    board: chess.Board,
    is_white: bool,
    worst_reward: float,
    expected_reward: float,
) -> None:
    """Test `engine_eval`."""
    # Eval
    r = engine_eval(
        engine_executable,
        board=board,
        is_white=is_white,
        worst_reward=worst_reward,
        time=5,
        depth=5,
    )
    # Tests
    _tests(r, is_white, expected_reward)


@pytest.mark.parametrize(
    "board, is_white, worst_reward, expected_reward",
    [
        (MATE_IN_ONE_FOR_WHITE, False, -1000, -1000),
        (MATE_IN_ONE_FOR_WHITE, True, -100, 100),
        (chess.Board(), False, -100, -50),
        (chess.Board(), True, -100, 44),
        (MATE_IN_ONE_FOR_WHITE.mirror(), True, -100, -100),
    ],
)
def test_engine_eval_from_engine(
    engine: SimpleEngine,
    board: chess.Board,
    is_white: bool,
    worst_reward: float,
    expected_reward: float,
) -> None:
    """Test `engine_eval`."""
    r = engine_eval(
        engine,
        board=board,
        is_white=is_white,
        worst_reward=worst_reward,
        time=5,
        depth=5,
    )
    # Tests
    _tests(r, is_white, expected_reward)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
