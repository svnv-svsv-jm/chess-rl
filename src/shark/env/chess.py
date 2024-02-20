__all__ = ["ChessEnv"]

from typing import Callable, Optional
from loguru import logger
import typing as ty
import os

import chess
from chess.engine import SimpleEngine, PovScore
import torch
from torch import Tensor
import random
from pathlib import Path

from tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    BinaryDiscreteTensorSpec,
    DiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
)
from matplotlib import pyplot as plt

from shark.utils import (
    board_to_tensor,
    move_action_space,
    action_one_hot_to_uci,
    action_to_one_hot,
    remove_illegal_move,
)
from shark.utils.patch import EnvBase

WORST_REWARD = -1e3


class ChessEnv(EnvBase):
    """Chess environment."""

    def __init__(
        self,
        engine_path: str = None,
        time: float = 5,
        depth: int = 20,
        flatten_state: bool = False,
        play_as: str = "white",
        play_vs_engine: bool = True,
        mate_amplifier: float = 10,
        softmax: bool = False,
        worst_reward: float = WORST_REWARD,
        illegal_amplifier: float = 100,
        lose_on_illegal_move: bool = True,
        use_one_hot: bool = True,
        from_engine: bool = False,
        **kwargs: ty.Any,
    ) -> None:
        """
        Args:
            engine_path (str):
                Path to chess engine. This class needs a usable chess engine.
                For example: `stockfish`.
                If not passed, this class will read from the `CHESS_ENGINE_EXECUTABLE` environment variable.
                If not set, an error will be raised.
                Please make sure to install a chess engine like Stockfish, and pass the correct
                installation path here.

            time (float, optional):
                Timeout value in seconds for engine.
                When the chess engine is called to validate a position or play a move, this will be the timeout for that.
                Defaults to `5`.

            depth (int, optional):
                Depth value for chess engine. When the chess engine is called to validate a position or play a move, this will be depth of the search tree.
                Defaults to `20`.

            flatten_state (bool, optional):
                Whether to flatten or not the state tensor.
                The state tensor has shape `(8, 8, 13)`, where 8 x 8 is the chess board, and for each square
                we have 13 possible values: 6 possible white pieces, 6 possible black pieces, empty square.
                If `True`, this `(8, 8, 13)` tensor will be flattened to `(8 * 8 * 13,)`.
                Consider not doing this.
                Defaults to `False`.

            play_as (str, optional):
                White or black. Defaults to `"white"`.

            play_vs_engine (bool, optional):
                Defaults to `True`. The `False` value is reserved for future use. Currently unsupported.
                In the future, you will be able to play against another RL agent.

            mate_amplifier (float, optional):
                Reward amplifier when mate happens.
                Defaults to `100`.

            softmax (bool, optional):
                If `True`, action values go through a softmax operation.
                This is useful, yet not necessary, when your agent outputs logits.
                Defaults to `False`.

            worst_reward (float, optional):
                Value for the worst reward possible, e.g. when losing a game. Defaults to `WORST_REWARD`.

            illegal_amplifier (float, optional):
                Reward amplifier when an illegal move is selected.
                Defaults to `1000`.

            lose_on_illegal_move (bool, optional): Defaults to `True`.
                Whether agent loses when an illegal move is played.
                If `False`, given that the action tensor is a vector of probabilities,
                the first legal action value will be enforced from it,
                giving priority to actions with higher probability.
                If `True`, the game is automatically lost when an illegal move is selected.
                Use this in combination with the `illegal_amplifier` parameter.

            use_one_hot (bool, optional):
                Whether to use one-hot vectors for state.
                Currently, this must be `True`, as `False` is not yet supported.
                Defaults to `True`.

            from_engine (bool, optional):
                Whetehr to sample moves from engine or randomly in the `sample()` method.
                The `sample()` method is not actively used when playing, only when doing rollouts.
                This is a rather irrelevant parameter.
        """
        super().__init__(**kwargs)  # call the constructor of the base class
        # Chess
        self.use_one_hot = use_one_hot
        self.illegal_amplifier = illegal_amplifier
        self.worst_reward = worst_reward
        self.softmax = softmax
        self.from_engine = from_engine
        if engine_path is None:
            engine_path = os.environ.get("CHESS_ENGINE_EXECUTABLE", "stockfish")
        assert Path(engine_path).exists(), f"Chess engine not found at {engine_path}."
        self.engine_path = engine_path
        self.time = time
        self.depth = depth
        play_as = play_as.lower()
        assert play_as in ["white", "black"]
        self.play_as = play_as
        self.play_vs_engine = play_vs_engine
        self.mate_amplifier = mate_amplifier
        self.is_white = self.play_as in ["white"]
        self.flatten = flatten_state
        self.lose_on_illegal_move = lose_on_illegal_move
        self.board = chess.Board()
        state = board_to_tensor(self.board, flatten=False)
        self.n_states = int(state.size(-1))
        action_space, self.action_map = move_action_space()
        self.n_actions = int(action_space.size(-1))
        if self.use_one_hot:
            # Action is a one-hot tensor
            self.action_spec = OneHotDiscreteTensorSpec(
                n=self.n_actions,
                shape=(self.n_actions,),
                device=self.device,
                dtype=torch.float32,
            )
            # Observation space
            self._state = OneHotDiscreteTensorSpec(
                n=self.n_states,
                shape=(8, 8, self.n_states),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            raise NotImplementedError("You must use one-hot vectors.")
            # # Action is a one-hot tensor
            # self.action_spec = DiscreteTensorSpec(
            #     n=self.n_actions,
            #     shape=(),
            #     device=self.device,
            #     dtype=torch.float32,
            # )
            # # Observation space
            # self._state = DiscreteTensorSpec(
            #     n=self.n_states,
            #     shape=(8, 8),
            #     device=self.device,
            #     dtype=torch.float32,
            # )
        self.observation_spec = CompositeSpec(observation=self._state)
        # Unlimited reward space
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=torch.Size([1]),
            device=self.device,
            dtype=torch.float32,
        )
        # Done
        self.done_spec = BinaryDiscreteTensorSpec(
            n=1,
            shape=torch.Size([1]),
            device=self.device,
            dtype=torch.bool,
        )
        logger.debug(f"action_spec: {self.action_spec}")
        logger.debug(f"observation_spec: {self.observation_spec}")
        logger.debug(f"reward_spec: {self.reward_spec}")

    def _reset(self, tensordict: TensorDict = None, **kwargs: ty.Any) -> TensorDict:
        """The `_reset()` method potentialy takes in a `TensorDict` and some kwargs which may contain data used in the resetting of the environment and returns a new `TensorDict` with an initial observation of the environment.

        The output `TensorDict` has to be new because the input tensordict is immutable.

        Args:
            tensordict (TensorDict):
                Immutable input.

        Returns:
            TensorDict:
                Initial state.
        """
        logger.debug("Resetting environment.")
        # Get shape and device
        if tensordict is not None:
            shape = tensordict.shape
            device = tensordict.device
        else:
            shape = torch.Size()
            device = self.device
        # Sanity check, should not end up here
        if device is None:
            device = self.device
        # Init chessboard
        self.board = chess.Board()
        # If playing as BLACK, let opponent move before we return the reset state
        if not self.is_white:
            with SimpleEngine.popen_uci(self.engine_path) as engine:
                # Opponent's move
                self._opponent_move(engine)
        # Init new state and pack it up in a TensorDict
        state = (
            board_to_tensor(self.board, flatten=self.flatten)
            .to(self.observation_spec.dtype)
            .to(device)
        )
        if self.flatten:
            state = state.flatten()
        # Return new TensorDict
        td = TensorDict(
            {
                "observation": state,
                # "reward": torch.Tensor([0]).to(self.reward_spec.dtype).to(device),
                "done": self.board.is_game_over(),
            },
            batch_size=shape,
            device=device,
        )
        logger.trace(f"State reset: {td}")
        return td

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """The `_step()` method takes in a `TensorDict` from which it reads an action, applies the action and returns a new `TensorDict` containing the observation, reward and done signal for that timestep.

        Args:
            tensordict (TensorDict): _description_

        Returns:
            TensorDict: _description_
        """
        # Read action from input
        action: Tensor = tensordict["action"]
        logger.trace(f"Reading action: {action.size()}")
        device = action.device
        # Check if game is already over?
        if self.board.is_game_over():
            logger.warning("How TF did you end up here?")
            outcome = self.board.outcome()
            if outcome is None:
                r = self.worst_reward
            else:
                if outcome.winner == self.is_white:
                    r = -self.worst_reward
                else:
                    r = self.worst_reward
                if self.board.is_checkmate():
                    logger.debug(f"{self.board.outcome()}")
                    r = r * self.mate_amplifier
            reward = reward = torch.Tensor([r]).to(self.reward_spec.dtype)
            # Return new TensorDict
            td = TensorDict(
                {
                    "observation": self.update_state().to(device),
                    "reward": reward.to(device),
                    "done": True,
                },
                batch_size=torch.Size(),
                device=device,
            )
            logger.trace(f"Returning new TensorDict: {td}")
            return td
        # Sanity check that should be impossible to violate
        assert (
            len(list(self.board.legal_moves)) > 0
        ), f"No legal move to choose from: {self.board.outcome()}"
        # Convert action to one-hot
        action = self._action_to_one_hot(action)
        # Get UCI
        uci = action_one_hot_to_uci(action)
        move = chess.Move.from_uci(uci)
        is_legal = self.board.is_legal(move)
        logger.trace(f"Requested action {move} (legal={is_legal})")
        # Softmax to have all positives
        if self.softmax:
            action = action.softmax(-1)
        # Remove illegal moves
        if not is_legal:
            logger.trace(f"Legalizing move {move}")
            action = remove_illegal_move(action, self.board, device=device)
            # Get action and its UCI
            # Action is a probability distribution over the action space
            uci = action_one_hot_to_uci(action)
            move = chess.Move.from_uci(uci)
            logger.trace(f"Chosen new (legal) move {move}")
        # Check if legal
        is_legal = self.board.is_legal(move)
        if not is_legal and self.lose_on_illegal_move:
            logger.trace(f"Illegal move {move}, returning very bad reward")
            state: torch.Tensor = tensordict["observation"]
            state = state.to(self.observation_spec.dtype)
            reward = torch.Tensor([self.worst_reward * self.illegal_amplifier])
            reward = reward.to(self.reward_spec.dtype)
            td = TensorDict(
                {
                    "observation": state.to(device),
                    "reward": reward.to(device),
                    "done": True,
                },
                batch_size=torch.Size(),
                device=device,
            )
            logger.trace(f"Returning {td}")
            return td
        assert is_legal, f"Illegal move: {uci}"
        # Apply move
        logger.trace(f"Pushing {uci}")
        self.board.push_san(uci)
        # Get evaliation from engine
        # If game not over, also let engine play
        with SimpleEngine.popen_uci(self.engine_path) as engine:
            # Get the evaluation from engine
            r = self._engine_eval(engine)
            # Check if game is over
            # If we won, set high reward
            # If game is over but we didn't win, it is stalemate or draw, return 0.0
            # If game not over, let opponet play
            if self.board.is_game_over():
                # Check if we won
                if self.board.is_checkmate():
                    logger.success(f"Game won!")
                    r = r * self.mate_amplifier
                else:
                    # We haven't won but game is over
                    logger.debug(f"{self.board.outcome()}")
                    r = 0.0
            else:
                # Opponent's move
                self._opponent_move(engine)
        # Reward
        if self.board.is_checkmate():
            logger.debug(f"Game lost!")
            r = r * self.mate_amplifier
        logger.trace(f"Reward: {r}")
        reward = torch.Tensor([r]).to(self.reward_spec.dtype)
        # Check if done
        done = self.board.is_game_over()
        # Update state
        state = self.update_state()
        # Return new TensorDict
        td = TensorDict(
            {
                "observation": state.to(device),
                "reward": reward.to(device),
                "done": done,
            },
            batch_size=torch.Size(),
            device=device,
        )
        logger.trace(f"Returning new TensorDict: {td}")
        return td

    def update_state(self) -> torch.Tensor:
        """Update state."""
        state = board_to_tensor(self.board, flatten=self.flatten).to(self.observation_spec.dtype)
        state = self._one_hot_state_to_discrete(state)
        if self.flatten:
            state = state.flatten()
        return state

    def sample(self, from_engine: bool = None) -> ty.Optional[TensorDict]:
        """Sample a legal move."""
        if from_engine is None:
            from_engine = self.from_engine
        if from_engine:
            # Get move rom engine
            with SimpleEngine.popen_uci(self.engine_path) as engine:
                logger.trace(f"Sampling a move from {self.engine_path}")
                result = engine.play(self.board, chess.engine.Limit(time=self.time))
                move = result.move
                if move is None:
                    logger.warning("No legal move by engine...")
                    return None
        else:
            # Get random move
            logger.trace("Getting a random move from legal moves.")
            legal_moves = list(self.board.legal_moves)
            if len(legal_moves) < 1:
                logger.warning("No legal move...")
                return None
            move = random.choice(legal_moves)
        # Get action tensor
        action = action_to_one_hot(move.uci(), chess_board=self.board)
        action = self._one_hot_action_to_discrete(action)
        action = action.to(self.action_spec.dtype)
        # Return TensorDict
        td = TensorDict(
            {"action": action.to(self.device)},
            batch_size=torch.Size(),
            device=self.device,
        )
        return td

    def _engine_eval(
        self,
        engine: SimpleEngine,
        board: chess.Board = None,
    ) -> float:
        """Let engine evaluate the current position."""
        if board is None:
            board = self.board
        logger.trace(f"Evaluating position with {self.engine_path}")
        info = engine.analyse(board, chess.engine.Limit(time=self.time, depth=self.depth))
        score: PovScore = info["score"]
        s = score.white() if self.is_white else score.black()
        r: ty.Optional[ty.Union[float, int]]
        r = (s).score()
        logger.trace(f"Score for position: {r}")
        if r is None:
            r = self.worst_reward
            # raise ValueError(f"Score: {s}")
        return r

    def _engine_move(
        self,
        engine: SimpleEngine,
        board: chess.Board = None,
    ) -> None:
        """Get opponent's move from engine."""
        if board is None:
            board = self.board
        logger.trace(f"Getting move from {self.engine_path}")
        result = engine.play(
            board,
            chess.engine.Limit(time=self.time, depth=self.depth),
        )
        move = result.move
        assert move is not None
        uci = move.uci()
        logger.trace(f"Engine's move: {uci}")
        board.push_san(uci)

    def _opponent_move(
        self,
        engine: SimpleEngine = None,
        board: chess.Board = None,
    ) -> None:
        if board is None:
            board = self.board
        if not board.is_game_over():
            logger.trace("Opponent's move")
            if self.play_vs_engine:
                assert engine is not None
                self._engine_move(engine)
            else:
                raise NotImplementedError("Only playing against an engine is supported.")

    def fake_tensordict(self) -> TensorDictBase:
        """Returns a fake `TensorDict` with key-value pairs that match in shape, device and dtype what can be expected during an environment rollout."""
        td: TensorDictBase = super().fake_tensordict()
        logger.debug(f"fake_tensordict: {td}")
        return td

    def rand_action(self, tensordict: Optional[TensorDictBase] = None) -> TensorDict:
        """Performs a random action given the `action_spec` attribute.

        Args:
            tensordict (TensorDictBase, optional):
                `TensorDict` object where the resulting action should be written.

        Returns:
            A tensordict object with the "action" entry updated with a random sample from the action-spec.
        """
        out = self.sample()
        if tensordict is not None:
            tensordict = tensordict.update(out, inplace=False)
        else:
            tensordict = out
        return tensordict

    # def _check_pawn_promotion(self, move: chess.Move) -> None:
    #     """Pawn promotion has an extra letter in UCI format."""
    #     if move not in self.board.legal_moves:
    #         move = chess.Move.from_uci(move.uci() + "q")  # assume promotion to queen
    #         if move not in self.board.legal_moves:
    #             raise ValueError(f"{move} not in {list(self.board.legal_moves)}")

    def _set_seed(self, seed: int) -> None:
        """The `_set_seed()` method sets the seed of any random number generator in the environment.

        Here we don't use any randomness but you can imagine a scenario where we initialize the state to a random value or add noise to the output observation in which case setting the random seed for reproducibility purposes would be very helpfull.

        Args:
            seed (int):
                Seed for RNG.
        """

    def _action_to_one_hot(self, action: torch.Tensor) -> torch.Tensor:
        """Convert action to one-hot if necessary. Always call this.

        Args:
            action (torch.Tensor): LongTensor.

        Returns:
            torch.Tensor: One-hot tensor.
        """
        # Convert action to one-hot
        if isinstance(self.action_spec, DiscreteTensorSpec):
            eye = torch.eye(self.n_actions).to(action.device)
            action = eye[action.long()]
        return action.float()

    def _one_hot_action_to_discrete(self, action: torch.Tensor) -> torch.Tensor:
        """Convert one-hot action to discrete if necessary. Always call this.

        Args:
            action (torch.Tensor): LongTensor.

        Returns:
            torch.Tensor: One-hot tensor.
        """
        # Convert action to one-hot
        if isinstance(self.action_spec, DiscreteTensorSpec):
            action = action.argmax(-1)
        return action.float()

    def _state_to_one_hot(self, state: torch.Tensor) -> torch.Tensor:
        """Convert state to one-hot if necessary. Always call this.

        Args:
            state (torch.Tensor): LongTensor.

        Returns:
            torch.Tensor: One-hot tensor.
        """
        # Convert action to one-hot
        if isinstance(self._state, DiscreteTensorSpec):
            eye = torch.eye(self.n_states).to(state.device)
            state = eye[state.long()]
        return state.float()

    def _one_hot_state_to_discrete(self, state: torch.Tensor) -> torch.Tensor:
        """Convert one-hot state to discrete if necessary. Always call this.

        Args:
            state (torch.Tensor): LongTensor.

        Returns:
            torch.Tensor: One-hot tensor.
        """
        # Convert state to one-hot
        if isinstance(self._state, DiscreteTensorSpec):
            state = state.argmax(-1)
        return state.float()

    def is_game_over(self) -> bool:
        """Tells you whether the game is over or not."""
        return self.board.is_game_over()
