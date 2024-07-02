__all__ = ["board_to_tensor"]

from loguru import logger
import chess
import torch
from torch import Tensor

from .const import FREE_SQUARE, PIECES_DICT, N_PIECES


def board_to_tensor(board: chess.Board, flatten: bool, one_hot: bool = True) -> Tensor:
    """Converts current board to a Tensor.

    Args:
        board (chess.Board):
            Chess board to convert to tensor.

        flatten (bool):
            Whether to flatten the board tensor or not.

        one_hot (bool, optional):
            Whether to return a discrete or one-hot representation.
            Defaults to `False`.

    Returns:
        Tensor:
            Tensor representation of the chess board.
            In each 8x8 square, there are 12 possible pieces or it is an empty square.
            The shape of this tensor is `(8,8,13)` if `flatten==False`, else it will be `(8*8*13,)`.
    """
    board_tensor = torch.zeros((8, 8, N_PIECES)).long()
    for i in range(8):
        for j in range(8):
            square = chess.square(i, j)
            piece = board.piece_at(square)
            # Fill in board tensor with correct values
            if piece is not None:
                piece_symbol = piece.symbol()
                board_tensor[i, j, PIECES_DICT[piece_symbol]] = 1
            else:
                board_tensor[i, j, PIECES_DICT[FREE_SQUARE]] = 1
    if not one_hot:
        board_tensor = board_tensor.argmax(-1)
    if flatten:
        board_tensor = board_tensor.flatten()
    return board_tensor
