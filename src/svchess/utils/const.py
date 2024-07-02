import typing as ty
from torch import Tensor

FREE_SQUARE: ty.Final[str] = "free"
PAWN_BLACK: ty.Final[str] = "p"
PAWN_WHITE: ty.Final[str] = "P"
KING_BLACK: ty.Final[str] = "k"
KING_WHITE: ty.Final[str] = "K"
ROOK_BLACK: ty.Final[str] = "r"
ROOK_WHITE: ty.Final[str] = "R"
BISHOP_BLACK: ty.Final[str] = "b"
BISHOP_WHITE: ty.Final[str] = "B"
QUEEN_BLACK: ty.Final[str] = "q"
QUEEN_WHITE: ty.Final[str] = "Q"
KNIGHT_BLACK: ty.Final[str] = "n"
KNIGHT_WHITE: ty.Final[str] = "N"


PIECES_DICT: ty.Final[ty.Dict[str, int]] = {
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

N_PIECES: ty.Final[int] = len(list(PIECES_DICT.values()))

N_ACTIONS: ty.Final[int] = 8 * 8

INITIAL_STATE: ty.Final[Tensor] = Tensor(
    [
        [8, 7, 0, 0, 0, 0, 1, 2],
        [9, 7, 0, 0, 0, 0, 1, 3],
        [10, 7, 0, 0, 0, 0, 1, 4],
        [11, 7, 0, 0, 0, 0, 1, 5],
        [12, 7, 0, 0, 0, 0, 1, 6],
        [10, 7, 0, 0, 0, 0, 1, 4],
        [9, 7, 0, 0, 0, 0, 1, 3],
        [8, 7, 0, 0, 0, 0, 1, 2],
    ]
)
