__all__ = ["board_to_tensor"]

from loguru import logger
import chess
import torch
from torch import Tensor


def board_to_tensor(board: chess.Board) -> Tensor:
    """Converts current board to a Tensor."""

    piece_dict = {
        "p": 0,
        "r": 1,
        "n": 2,
        "b": 3,
        "q": 4,
        "k": 5,
        "P": 6,
        "R": 7,
        "N": 8,
        "B": 9,
        "Q": 10,
        "K": 11,
    }

    tensor = torch.zeros((8, 8, len(list(piece_dict.values())))).long()

    for i in range(8):
        for j in range(8):
            square = chess.square(i, j)
            piece = board.piece_at(square)

            if piece is not None:
                piece_type = piece.symbol()
                logger.trace(f"Piece type ({i},{j}): {piece_type}")
                tensor[i, j, piece_dict[piece_type]] = 1

    return tensor
