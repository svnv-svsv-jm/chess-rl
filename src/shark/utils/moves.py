__all__ = [
    "action_to_one_hot_legal",
    "action_to_one_hot",
    "move_action_space",
    "action_one_hot_to_uci",
    "action_to_uci",
    "get_move_score",
    "remove_illegal_move",
    "uci_to_one_hot_action",
]

import typing as ty
from loguru import logger

import chess
from chess.engine import SimpleEngine, PovScore
import torch


def uci_to_one_hot_action(uci: str) -> torch.Tensor:
    """Transforms a UCI to a one-hot action vector."""
    a, action_map = move_action_space()
    idx = 0
    for uci_, idx in action_map.items():
        if uci == uci_:
            break
    a[idx] = 1
    return a


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


def action_to_uci(action: torch.Tensor) -> str:
    """Decode a given tensor into a move UCI.

    Args:
        action (torch.Tensor): ()
            Action. Should be an int.

    Returns:
        str:
            Move UCI.
    """
    _, action_dict = move_action_space()
    idx = action.view(-1).argmax()
    moves = list(action_dict.keys())
    uci = moves[idx]
    logger.trace(f"Getting move from index {idx} out of {len(moves)} moves: {uci}")
    return uci


def action_one_hot_to_uci(action_one_hot: torch.Tensor) -> str:
    """Decode a given one-hot tensor into a move UCI.

    Args:
        action_one_hot (torch.Tensor): (N,)
            Move one-hot tensor.

    Returns:
        str:
            Move UCI.
    """
    n_act_ = int(action_one_hot.size(-1))
    action_space, action_dict = move_action_space()
    n_act = int(action_space.numel())
    assert (
        n_act_ == n_act
    ), f"Action space has size {action_space.size()}, but the input action vector has size {action_one_hot.size()}"
    idx = action_one_hot.view(-1).argmax()
    moves = list(action_dict.keys())
    try:
        uci = moves[idx]
    except IndexError as ex:
        raise IndexError(f"Index {idx} out of range {len(moves)}") from ex
    logger.trace(f"Getting move from index {idx} out of {len(moves)} moves: {uci}")
    return uci


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
            if from_square == to_square:
                continue  # Can't go from square to same square...
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


def remove_illegal_move(
    action: torch.Tensor,
    board: chess.Board,
    device: torch.device = None,
) -> torch.Tensor:
    """Remove illegal moves from the actions."""
    assert len(list(board.legal_moves)) > 0, f"No legal move to choose from: {board.outcome()}"
    # Check device
    if device is None:
        device = action.device
    # Get empty action vector and all moves
    a, act_dict = move_action_space()
    # Create mask, set to True when move is legal
    mask = torch.zeros_like(action, device=device).bool()
    for uci, i in act_dict.items():
        move = chess.Move.from_uci(uci)
        if board.is_legal(move):
            mask[i] = True
    # Unsqueeze bullshit
    if mask.dim() < action.dim():
        mask = mask.unsqueeze(0)
    # Check at least one legal move exists
    assert mask.sum() > 0
    # Find maximum among legal values
    possible_values = action[mask]
    # Get index
    max_index = possible_values.argmax(-1).to(device)
    i_tensor = torch.arange(action.size(-1)).to(device)
    indeces = i_tensor[mask.to(device)].to(device)
    actual_index = indeces[max_index]
    # Return that move
    a[actual_index] = 1
    return a.to(device)
