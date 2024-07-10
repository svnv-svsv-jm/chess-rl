import pytest
from loguru import logger
import typing as ty
import sys

import chess
from chess.engine import SimpleEngine

from svchess.utils import check_winner, on_gameover, play_move

# Create a board with a mate in one board for white
MATE_IN_ONE_FOR_WHITE = chess.Board("5Q2/5K1k/8/8/8/8/8/8 w - - 0 1")
# Draw
DRAW = chess.Board("7k/5Q2/7K/8/8/8/8/8 b - - 0 1")


@pytest.mark.parametrize("board", [MATE_IN_ONE_FOR_WHITE, MATE_IN_ONE_FOR_WHITE.mirror(), DRAW])
def test_check_winner(engine: SimpleEngine, board: chess.Board) -> None:
    """Test `check_winner`."""
    over, winner = check_winner(board)
    logger.info(f"Over: {over}")
    logger.info(f"Winner: {winner}")
    reward = on_gameover(board, play_as=True, highest_reward=1000, on_draw=100)
    if over:
        assert reward == 100
    else:
        assert reward is None
    if not over:
        board = play_move(board, engine=engine)
        over, winner = check_winner(board)
        logger.info(f"Over: {over}")
        logger.info(f"Winner: {winner}")
        if isinstance(winner, bool):
            reward = on_gameover(board, play_as=winner, highest_reward=1000, on_draw=100)
            logger.info(f"Reward: {reward}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
