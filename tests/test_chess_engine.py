import pytest
from loguru import logger
import typing as ty
import sys

import chess
from chess.engine import SimpleEngine


def test_call_chess_engine(engine: SimpleEngine) -> None:
    """Test we can call chess engine."""
    # Call engine
    analysed_variations = engine.analyse(
        chess.Board(),
        limit=chess.engine.Limit(depth=18),
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


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
