__all__ = ["ChessEnv"]

from typing import Callable, Optional
from loguru import logger
import typing as ty

import chess
from chess.engine import SimpleEngine, PovScore
import torch
from torch import Tensor
import random

from tensordict import TensorDict, TensorDictBase
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import EnvBase
from torchrl.envs.utils import set_exploration_mode

from matplotlib import pyplot as plt

from shark.utils import board_to_tensor, move_action_space, action_one_hot_to_uci, action_to_one_hot


class ChessEnv(EnvBase):
    """Chess environment."""

    def __init__(
        self,
        engine_path: str,
        time: float = 5,
        depth: int = 20,
        play_as: str = "white",
        play_vs_engine: bool = True,
        mate_amplifier: float = 100,
        softmax: bool = True,
    ) -> None:
        """_summary_

        Args:
            engine_path (str): _description_
            time (float, optional): _description_. Defaults to 5.
            depth (int, optional): _description_. Defaults to 20.
            play_as (str, optional): _description_. Defaults to "white".
            play_vs_engine (bool, optional): _description_. Defaults to True.
            mate_amplifier (float, optional): _description_. Defaults to 100.
        """
        super().__init__()  # call the constructor of the base class
        # Chess
        self.softmax = softmax
        self.engine_path = engine_path
        self.time = time
        self.depth = depth
        play_as = play_as.lower()
        assert play_as in ["white", "black"]
        self.play_as = play_as
        self.play_vs_engine = play_vs_engine
        self.mate_amplifier = mate_amplifier
        self.is_white = self.play_as in ["white"]
        self.board = chess.Board()
        self.state = board_to_tensor(self.board)
        self.action_space, self.action_map = move_action_space()
        self.dtype = torch.float32

        # action_spec, observation_spec and reward_spec are fields that define the range and shape of our actions, observations and rewards. They tell other modules in the library what to expect from the environment. They are some type of TensorSpec depending on whether they're bounded or not, discrete or continuous. observation_spec is a bit of an outlier - it has to be of type CompositeSpec.

        # Action is a one-hot tensor
        self.action_spec = BoundedTensorSpec(
            minimum=0,
            maximum=1,
            shape=self.action_space.size(),
            dtype=self.dtype,
        )

        # Observation space: it is a 8x8x13 box
        observation_spec = BoundedTensorSpec(
            low=0,
            high=1,
            shape=self.state.size(),
            dtype=self.dtype,
        )
        # has to be CompositeSpec (not sure why)
        self.observation_spec = CompositeSpec(observation=observation_spec)

        # Unlimited reward space
        self.reward_spec = UnboundedContinuousTensorSpec(shape=torch.Size([1]), dtype=torch.float32)

    def rand_action(self, tensordict: Optional[TensorDictBase] = None) -> TensorDict:
        """Performs a random action given the action_spec attribute.

        Args:
            tensordict (TensorDictBase, optional): tensordict where the resulting action should be written.

        Returns:
            a tensordict object with the "action" entry updated with a random
            sample from the action-spec.
        """
        out = self.sample()
        if tensordict is not None:
            tensordict.update(out)
        else:
            tensordict = out
        return tensordict

    def _reset(self, tensordict: TensorDict = None, **kwargs: ty.Any) -> TensorDict:
        """The `_reset()` method potentialy takes in a `TensorDict` and some kwargs which may contain data used in the resetting of the environment and returns a new `TensorDict` with an initial observation of the environment.

        The output `TensorDict` has to be new because the input tensordict is immutable. In this example we don't use the input and we just set the inital state to `0`, but you can imagine a situation where you could use the input to initialize a state or generate a random state.

        Args:
            tensordict (TensorDict): _description_

        Returns:
            TensorDict: _description_
        """
        logger.trace(tensordict)
        if tensordict is not None:
            shape = tensordict.shape
        else:
            shape = torch.Size()

        self.board = chess.Board()

        # If playing as BLACK, let opponent move before we return the reset state
        if not self.is_white:
            with SimpleEngine.popen_uci(self.engine_path) as engine:
                # Opponent's move
                self._opponent_move(engine)

        # init new state and pack it up in a tensordict
        self.state = board_to_tensor(self.board).to(self.device).to(self.dtype)
        logger.trace(f"State reset: {self.state.size()}")
        return TensorDict(
            {"observation": self.state},
            batch_size=shape,
        )

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """The `_step()` method takes in a `TensorDict` from which it reads an action, applies the action and returns a new `TensorDict` containing the observation, reward and done signal for that timestep.

        Args:
            tensordict (TensorDict): _description_

        Returns:
            TensorDict: _description_
        """
        # Read action from input
        action: Tensor = tensordict["action"]
        action = action.float()
        logger.trace(f"Action ({action.size()}): action")
        # Softmax to have all positives
        if self.softmax:
            action = action.softmax(-1)
        # Remove illegal moves
        mask = torch.zeros(self.action_space.size(), device=self.device).view(-1)
        _, act_dict = move_action_space()
        for uci, i in act_dict.items():
            move = chess.Move.from_uci(uci)
            if self.board.is_legal(move):
                mask[i] = 1
        if mask.dim() < action.dim():
            mask = mask.unsqueeze(0)
        action = (action * mask).float()
        # Get action and its UCI
        # Action is a probability distribution over the action space
        logger.trace(f"Action: {action}")
        uci = action_one_hot_to_uci(action)
        logger.trace(f"Move: {uci}")
        assert self.board.is_legal(chess.Move.from_uci(uci)), f"Illegal move: {uci}"

        with SimpleEngine.popen_uci(self.engine_path) as engine:
            # Move
            self.board.push_san(uci)
            # Get the evaluation from engine
            r = self._engine_eval(engine)
            # Opponent's move
            self._opponent_move(engine)

        # Reward
        logger.trace(f"Reward: {r}")
        reward = torch.Tensor([r]).float()

        # Check if done
        done = self.board.is_game_over()

        # Update state
        self.state = board_to_tensor(self.board).to(self.dtype)

        return TensorDict(
            {
                "observation": self.state.to(self.device),
                "reward": reward.to(self.device),
                "done": done,
            },
            batch_size=torch.Size(),
        )

    def _set_seed(self, seed: int) -> None:
        """The `_set_seed()` method sets the seed of any random number generator in the environment.

        Here we don't use any randomness but you can imagine a scenario where we initialize the state to a random value or add noise to the output observation in which case setting the random seed for reproducibility purposes would be very helpfull.

        Args:
            seed (int):
                Seed for RNG.
        """

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        logger.warning("No sense in calling this method.")
        return tensordict

    def set_state(self, *args: ty.Any, **kwargs: ty.Any) -> None:
        logger.warning("No sense in calling this method.")

    def sample(self, from_engine: bool = False) -> ty.Optional[TensorDict]:
        """Sample a legal move."""
        if from_engine:
            with SimpleEngine.popen_uci(self.engine_path) as engine:
                logger.trace(f"Sampling a move from {self.engine_path}")
                result = engine.play(self.board, chess.engine.Limit(time=self.time))
                move = result.move
                if move is None:
                    return None
        else:
            legal_moves = list(self.board.legal_moves)
            if len(legal_moves) < 1:
                return None
            move = random.choice(legal_moves)
        action = action_to_one_hot(move.uci(), chess_board=self.board).to(self.dtype)
        return TensorDict(
            {"action": action},
            batch_size=torch.Size(),
        )

    def _engine_eval(self, engine: SimpleEngine) -> float:
        """Let engine evaluate the position."""
        logger.trace(f"Evaluating position with {self.engine_path}")
        info = engine.analyse(self.board, chess.engine.Limit(time=self.time, depth=self.depth))
        score: PovScore = info["score"]
        s = score.white() if self.is_white else score.black()
        r: ty.Optional[ty.Union[float, int]]
        r = (s).score()
        if r is None:
            r = -1e6
            # raise ValueError(f"Score: {s}")
        if self.board.is_checkmate():
            r = r * self.mate_amplifier
        logger.trace(f"Score: {r}")
        return r

    def _engine_move(self, engine: SimpleEngine) -> None:
        """Get opponent's move from engine."""
        logger.trace(f"Getting move from {self.engine_path}")
        result = engine.play(
            self.board,
            chess.engine.Limit(time=self.time, depth=self.depth),
        )
        move = result.move
        assert move is not None
        uci = move.uci()
        logger.trace(f"Engine's move: {uci}")
        self.board.push_san(uci)

    def _opponent_move(self, engine: SimpleEngine = None) -> None:
        if not self.board.is_game_over():
            logger.trace("Opponent's move")
            if self.play_vs_engine:
                assert engine is not None
                self._engine_move(engine)
            else:
                raise NotImplementedError("")
