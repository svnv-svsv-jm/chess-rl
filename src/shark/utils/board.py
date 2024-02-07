__all__ = ["board_to_tensor"]

from loguru import logger
import chess
import torch
from torch import Tensor

from .const import (
    FREE_SQUARE,
    PAWN_BLACK,
    PAWN_WHITE,
    KING_BLACK,
    KING_WHITE,
    QUEEN_BLACK,
    QUEEN_WHITE,
    KNIGHT_BLACK,
    KNIGHT_WHITE,
    BISHOP_BLACK,
    BISHOP_WHITE,
    ROOK_BLACK,
    ROOK_WHITE,
)


def board_to_tensor(board: chess.Board, flatten: bool = True) -> Tensor:
    """Converts current board to a Tensor.

    Args:
        board (chess.Board):
            Chess board to convert to tensor.

    Returns:
        Tensor: (8,8,13)
            Tensor representation of the chess board. In each 8x8 square, there are 12 possible pieces or it is an empty square.

    """
    piece_dict = {
        FREE_SQUARE: 0,
        PAWN_BLACK: 1,
        ROOK_BLACK: 2,
        KNIGHT_BLACK: 3,
        BISHOP_BLACK: 4,
        QUEEN_BLACK: 5,
        KING_BLACK: 6,
        PAWN_WHITE: 7,
        ROOK_WHITE: 8,
        KNIGHT_WHITE: 9,
        BISHOP_WHITE: 10,
        QUEEN_WHITE: 11,
        KING_WHITE: 12,
    }

    tensor = torch.zeros((8, 8, len(list(piece_dict.values())))).long()

    for i in range(8):
        for j in range(8):
            square = chess.square(i, j)
            piece = board.piece_at(square)

            if piece is not None:
                piece_type = piece.symbol()
                tensor[i, j, piece_dict[piece_type]] = 1
            else:
                tensor[i, j, piece_dict[FREE_SQUARE]] = 1

    if flatten:
        tensor = tensor.flatten()

    return tensor
