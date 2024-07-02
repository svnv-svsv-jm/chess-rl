__all__ = ["get_random_move", "action_dict", "action_int_to_uci", "action_int_to_move"]

import typing as ty
from loguru import logger

from torch import Tensor
import random
import chess


def get_random_move(board: chess.Board) -> ty.Optional[chess.Move]:
    """Select random move."""
    logger.trace("Getting a random move from legal moves.")
    legal_moves = list(board.legal_moves)
    if len(legal_moves) < 1:  # pragma: no cover
        logger.warning("No legal move...")
        return None
    move = random.choice(legal_moves)
    return move


def action_dict() -> ty.Dict[str, int]:
    """Creates the action dictionary.

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
            if from_square == to_square:
                continue  # Can't go from square to same square...
            move_ = chess.Move(from_square, to_square)
            m: str = move_.uci()
            all_moves.append(m)
    # Get unique moves just in case
    unique_moves: ty.List[str] = list(set(all_moves))  # Remove duplicate moves if any
    # Here the action mapping
    action_dict: ty.Dict[str, int] = {move: i for i, move in enumerate(unique_moves)}
    # Return
    return action_dict


def action_int_to_uci(action: int | Tensor) -> str | None:
    """Converts input integer to move UCI."""
    a_dict = action_dict()
    for key, value in a_dict.items():
        if value == action:
            return key
    return None  # pragma: no cover


def action_int_to_move(action: int | Tensor) -> chess.Move:
    """Converts input integer to move."""
    move_uci = action_int_to_uci(action)
    if move_uci is None:  # pragma: no cover
        raise RuntimeError(f"Could not convert action: {action}.")
    move = chess.Move.from_uci(move_uci)
    return move
