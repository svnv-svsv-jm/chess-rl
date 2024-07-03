__all__ = ["engine_eval"]

import typing as ty
from loguru import logger
import chess
from chess.engine import SimpleEngine, PovScore


def engine_eval(
    engine: SimpleEngine | str,
    board: chess.Board,
    is_white: bool,
    worst_reward: float,
    **kwargs: ty.Any,
) -> float:
    """Let engine evaluate the current position and return it as reward.

    Args:
        engine (SimpleEngine | str):
            If `SimpleEngine`, then this is the engine that has to perform the evaluation.
            If `str`, then this is the path to the engine that will be opened as `SimpleEngine.popen_uci(engine_path)`.

        board (chess.Board):
            Current chessboard.

        is_white (bool):
            Whether we're playing as white or not.

        worst_reward (float):
            Worst possible reward, in centipawns.

        **kwargs (Any):
            Inputs for `chess.engine.Limit()`.

    Returns:
        float: Evaluation of current position, in centipawns.
    """
    if isinstance(engine, str):
        return _engin_eval_from_path(
            engine,
            board=board,
            is_white=is_white,
            worst_reward=worst_reward,
            **kwargs,
        )
    if isinstance(engine, SimpleEngine):
        return _engine_eval(
            engine,
            board=board,
            is_white=is_white,
            worst_reward=worst_reward,
            **kwargs,
        )
    raise ValueError(f"Engine must be a {str} or an instance of {SimpleEngine}.")


def _engine_eval(
    engine: SimpleEngine,
    board: chess.Board,
    is_white: bool,
    worst_reward: float,
    **kwargs: ty.Any,
) -> float:
    """Let engine evaluate the current position and return it as reward.

    Args:
        engine (SimpleEngine):
            Engine that has to perform the evaluation.

        board (chess.Board):
            Current chessboard.

        is_white (bool):
            Whether we're playing as white or not.

        worst_reward (float):
            Worst possible reward.

        **kwargs (Any):
            Inputs for `chess.engine.Limit()`.

    Returns:
        float: Evaluation of current position.
    """
    logger.trace(f"Evaluating position with {engine}")
    info = engine.analyse(board, chess.engine.Limit(**kwargs))
    pov_score: PovScore = info["score"]
    score = pov_score.white() if is_white else pov_score.black()
    r: ty.Optional[ty.Union[float, int]]
    mate_score = score.mate()
    centipawn_score = score.score()
    logger.trace(f"Position: centipawn_score={centipawn_score} | mate_score={mate_score}")
    if mate_score is None and centipawn_score is None:
        r = worst_reward
    elif mate_score is None and centipawn_score is not None:
        r = centipawn_score
    elif centipawn_score is None and mate_score is not None:
        r = worst_reward if mate_score < 0 else -worst_reward
    elif centipawn_score is not None and mate_score is not None:
        r = centipawn_score
    else:
        raise RuntimeError(
            f"Impossible to evaluate position: centipawn_score={centipawn_score} and mate_score={mate_score}"
        )
    return r


def _engin_eval_from_path(
    engine_path: str,
    board: chess.Board,
    is_white: bool,
    worst_reward: float,
    **kwargs: ty.Any,
) -> float:
    """Let engine evaluate the current position and return it as reward.

    Args:
        engine_path (str):
            Path to the engine that will be opened as `SimpleEngine.popen_uci(engine_path)`.

        board (chess.Board):
            Current chessboard.

        is_white (bool):
            Whether we're playing as white or not.

        worst_reward (float):
            Worst possible reward.

        **kwargs (Any):
            Inputs for `chess.engine.Limit()`.

    Returns:
        float: Evaluation of current position.
    """
    with SimpleEngine.popen_uci(engine_path) as engine:
        r = _engine_eval(
            engine,
            board=board,
            is_white=is_white,
            worst_reward=worst_reward,
            **kwargs,
        )
    return r
