__all__ = ["check_winner", "on_gameover"]

import typing as ty
from loguru import logger
import chess


def check_winner(board: chess.Board) -> ty.Tuple[bool, bool | None]:
    """Checks who's won.

    Args:
        board (chess.Board): Chess board.

    Returns:
        over (bool):
            Whether the game is over or not.
        winner (bool | None):
            `True` if white won, `False` otherwise. `None` if game is drawn or not over.
    """
    outcome = board.outcome()
    over = False
    if outcome:
        over = True
        if outcome.winner == chess.WHITE:
            return over, True
        if outcome.winner == chess.BLACK:
            return over, False
        return over, None
    return over, None


def on_gameover(
    board: chess.Board,
    play_as: bool,
    highest_reward: float,
    on_draw: float,
) -> float | None:
    """Decide the reward to return on gameovers.

    Args:
        board (chess.Board):
            Chess board.

        play_as (bool):
            If `True`, it's playing as white.

        highest_reward (float):
            Highest reward value.

        on_draw (float):
            Reward value on draw.

    Returns:
        reward (float | None): Reward.
    """
    over, winner = check_winner(board)
    if not over:
        return None
    if winner is None:  # Draw
        r = on_draw
    else:  # There is a winner
        r = highest_reward if winner == play_as else -highest_reward
    return r
