__all__ = ["play_move"]

import typing as ty
from loguru import logger
import chess
from chess.engine import SimpleEngine

from .moves import get_random_move


def _engine_play(board: chess.Board, engine: SimpleEngine, **kwargs: ty.Any) -> chess.Board:
    """Helper."""
    logger.trace(f"Playing move [{kwargs}]")
    result = engine.play(board, chess.engine.Limit(**kwargs))
    move = result.move
    assert move is not None
    board.push(move)
    return board


def play_move(
    board: chess.Board,
    engine_executable: str = None,
    engine: SimpleEngine = None,
    **kwargs: ty.Any,
) -> chess.Board:
    """Play move.

    Args:
        board (chess.Board):
            Chess board.

        engine_executable (str, optional):
            Path to chess engine. This class needs a usable chess engine.
            Defaults to `None`.

        **kwargs (Any):
            Parameters for `chess.engine.Limit(**kwargs)`.

    Returns:
        chess.Board: Chessboard, updated with a move.
    """
    # If engine is inputted
    if isinstance(engine, SimpleEngine):
        board = _engine_play(board, engine, **kwargs)

    # Open engine and push move
    elif isinstance(engine_executable, str):
        with SimpleEngine.popen_uci(engine_executable) as engine:
            board = _engine_play(board, engine, **kwargs)

    # Sample random move
    else:
        move = get_random_move(board)
        if move is None:  # pragma: no cover
            raise RuntimeError("Failed to sample a random move.")
        board.push(move)

    return board
