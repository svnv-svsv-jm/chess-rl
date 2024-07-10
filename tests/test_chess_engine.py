import pytest
from loguru import logger
import typing as ty
import sys

import chess
from chess.engine import SimpleEngine


def test_call_chess_engine(engine_executable: str) -> None:
    """Test we can call chess engine."""
    board = chess.Board()
    with SimpleEngine.popen_uci(engine_executable) as engine:
        # Call engine
        analysed_variations = engine.analyse(
            board,
            limit=chess.engine.Limit(time=5, depth=18),
            multipv=5,
        )
        logger.info(f"Analysis:\n{analysed_variations}")
        # Top 5 moves
        top_five_moves: ty.List[dict] = []
        for variation in analysed_variations:
            move = variation["pv"][0]
            score = variation["score"]
            top_five_moves.append(dict(move=move, score=score))
        logger.info(f"Top 5 moves: {top_five_moves}")


def test_play_game(engine_executable: str) -> None:
    """Test playing a game until the end."""
    board = chess.Board()
    n_move_counter = 0
    with SimpleEngine.popen_uci(engine_executable) as engine:
        while not board.is_game_over():
            result = engine.play(board, chess.engine.Limit(time=0.1))
            move = result.move
            assert move is not None
            board.push(move)
            n_move_counter += 1
            logger.debug(f"Played {n_move_counter} move(s).")
    logger.info(f"Game ended after {n_move_counter} moves because of {board.outcome()}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
