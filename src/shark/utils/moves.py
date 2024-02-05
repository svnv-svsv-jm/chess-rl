__all__ = [
    "action_to_one_hot_legal",
    "action_to_one_hot",
    "move_action_space",
    "action_one_hot_to_uci",
    "get_move_score",
]

import typing as ty
from loguru import logger

import chess
from chess.engine import SimpleEngine, PovScore
import torch
import numpy as np


def get_move_score(
    board: chess.Board,
    move: chess.Move,
    engine_path: str,
    time: float = None,
    depth: int = None,
) -> PovScore:
    """Returns a score for a potential move. The move is tried, scored and then undone.

    Args:
        board (chess.Board):
            Chess board.
        move (chess.Move):
            Potential move to score.
        engine_path (str):
            Path to the engine.
        time (float):
            Time limit for analysis.

    Returns:
        PovScore:
            Score for the move.
    """
    # Create a chess engine instance
    with SimpleEngine.popen_uci(engine_path) as engine:
        # Set up the position on the board
        board.push(move)

        # Get the evaluation from engine
        info = engine.analyse(board, chess.engine.Limit(time=time, depth=depth))
        logger.trace(f"Info: {info}")
        score: PovScore = info["score"]

        # Print the evaluation score
        logger.trace(f"Score for move {move.uci()}: {score}")

        # Undo the move to restore the original position
        board.pop()

    return score


def action_one_hot_to_uci(action_one_hot: torch.Tensor) -> str:
    """Decode a given one-hot tensor into a move UCI.

    Args:
        action_one_hot (torch.Tensor): (N,)
            Move one-hot tensor.

    Returns:
        str:
            Move UCI.
    """
    _, action_dict = move_action_space()
    idx = action_one_hot.argmax()
    return list(action_dict.keys())[idx]


def action_to_one_hot(move: str, chess_board: chess.Board) -> torch.Tensor:
    """Encode a given move UCI into a one-hot tensor.

    Args:
        move (str):
            Move UCI we want to encode.
        chess_board (chess.Board):
            Chess board object.

    Returns:
        torch.Tensor:
            One-hot tensor.
    """
    action_vec, action_dict = move_action_space()
    action_space_size = action_vec.numel()
    action_one_hot = torch.zeros(action_space_size, dtype=torch.int8)

    move_index = action_dict[move]
    action_one_hot[move_index] = 1

    return action_one_hot


def action_to_one_hot_legal(move: str, chess_board: chess.Board) -> torch.Tensor:
    """Encode a given move UCI into a one-hot tensor. The size of this tensor depends on the number of possible legal moves.

    Args:
        move (str):
            Move UCI we want to encode.
        chess_board (chess.Board):
            Chess board object.

    Returns:
        torch.Tensor:
            One-hot tensor. The size of this tensor depends on the number of possible legal moves.
    """
    legal_moves = [move.uci() for move in chess_board.legal_moves]
    logger.trace(f"Legal moves: {legal_moves}")
    action_space_size = len(legal_moves)
    action_one_hot = torch.zeros(action_space_size, dtype=torch.int8)

    move_index = legal_moves.index(move)
    action_one_hot[move_index] = 1

    return action_one_hot


def move_action_space() -> ty.Tuple[torch.Tensor, ty.Dict[str, int]]:
    """Create a one-hot tensor of all possible moves and the action dictionary.

    Returns:
        torch.Tensor: (N,)
            Action space vector.
        ty.Dict[str, int]:
            Action dictionary, mapping move UCI to index.
    """
    # Get all moves
    all_moves: ty.List[str] = []
    for from_square in chess.SQUARES:
        for to_square in chess.SQUARES:
            move_ = chess.Move(from_square, to_square)
            m: str = move_.uci()
            all_moves.append(m)
    # Get unique moves just in case
    unique_moves: ty.List[str] = list(set(all_moves))  # Remove duplicate moves if any
    # Here the action mapping
    action_dict: ty.Dict[str, int] = {move: i for i, move in enumerate(unique_moves)}
    # Action space vector
    action_space_size: int = len(unique_moves)
    logger.trace(f"Action space size: {action_space_size}")
    action_space = torch.zeros((action_space_size,), dtype=torch.int8)
    # Return
    return action_space, action_dict
