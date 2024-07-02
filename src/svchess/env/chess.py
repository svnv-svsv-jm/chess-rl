__all__ = ["Chess"]

import typing as ty
from loguru import logger

import os
from pathlib import Path
import chess
from chess.engine import SimpleEngine, PovScore
import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    BinaryDiscreteTensorSpec,
    DiscreteTensorSpec,
)

from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp

from svchess.utils.const import N_PIECES, N_ACTIONS
from svchess.utils.moves import get_random_move, action_dict
from .utils import make_specs


class Chess(EnvBase):
    """Chess RL environment."""

    def __init__(
        self,
        engine_path: str = None,
        timeout: float = 5,
        device: torch.device | str | int | None = None,
        batch_size: torch.Size | None = None,
        run_type_checks: bool = True,
        allow_done_after_reset: bool = False,
    ):
        """
        Args:
            engine_path (str):
                Path to chess engine. This class needs a usable chess engine.
                For example: `stockfish`.
                If not passed, this class will read from the `CHESS_ENGINE_EXECUTABLE` environment variable.
                If not set, a warning will be raised.
                Please make sure to install a chess engine like Stockfish, and pass the correct installation path here.

            timeout (float, optional):
                Timeout value in seconds for engine.
                When the chess engine is called to validate a position or play a move, this will be the timeout for that.
                Defaults to `5`.

            device (torch.device): The device of the environment. Deviceless environments
                are allowed (device=None). If not `None`, all specs will be cast
                on that device and it is expected that all inputs and outputs will
                live on that device.
                Defaults to `None`.

            batch_size (torch.Size or equivalent, optional): batch-size of the environment.
                Corresponds to the leading dimension of all the input and output
                tensordicts the environment reads and writes. Defaults to an empty batch-size.

            run_type_checks (bool, optional): If `True`, type-checks will occur
                at every reset and every step. Defaults to `False`.

            allow_done_after_reset (bool, optional): if `True`, an environment can
                be done after a call to :meth:`~.reset` is made. Defaults to `False`.

        Attributes:
            done_spec (CompositeSpec): equivalent to `full_done_spec` as all
                `done_specs` contain at least a `"done"` and a `"terminated"` entry

            action_spec (TensorSpec): the spec of the action. Links to the spec of the leaf
                action if only one action tensor is to be expected. Otherwise links to
                `full_action_spec`.

            observation_spec (CompositeSpec): equivalent to `full_observation_spec`.

            reward_spec (TensorSpec): the spec of the reward. Links to the spec of the leaf
                reward if only one reward tensor is to be expected. Otherwise links to
                `full_reward_spec`.

            state_spec (CompositeSpec): equivalent to `full_state_spec`.

            full_done_spec (CompositeSpec): a composite spec such that `full_done_spec.zero()`
                returns a tensordict containing only the leaves encoding the done status of the
                environment.

            full_action_spec (CompositeSpec): a composite spec such that `full_action_spec.zero()`
                returns a tensordict containing only the leaves encoding the action of the
                environment.

            full_observation_spec (CompositeSpec): a composite spec such that `full_observation_spec.zero()`
                returns a tensordict containing only the leaves encoding the observation of the
                environment.

            full_reward_spec (CompositeSpec): a composite spec such that `full_reward_spec.zero()`
                returns a tensordict containing only the leaves encoding the reward of the
                environment.

            full_state_spec (CompositeSpec): a composite spec such that `full_state_spec.zero()`
                returns a tensordict containing only the leaves encoding the inputs (actions
                excluded) of the environment.

            batch_size (torch.Size): The batch-size of the environment.

            device (torch.device): the device where the input/outputs of the environment
                are to be expected. Can be `None`.

        Methods:
            step (TensorDictBase -> TensorDictBase): step in the environment

            reset (TensorDictBase, optional -> TensorDictBase): reset the environment

            set_seed (int -> int): sets the seed of the environment

            rand_step (TensorDictBase, optional -> TensorDictBase): random step given the action spec

            rollout (Callable, ... -> TensorDictBase):
                Executes a rollout in the environment with the given policy (or random steps if no policy is provided)
        """
        super().__init__(
            device=device,
            batch_size=batch_size,
            run_type_checks=run_type_checks,
            allow_done_after_reset=allow_done_after_reset,
        )

        # Attributes from inputs
        if engine_path is None:
            engine_path = os.environ.get("CHESS_ENGINE_EXECUTABLE", "stockfish")
        if not Path(engine_path).exists():
            logger.warning(f"Chess engine not found at {engine_path}.")
        self.engine_path = engine_path
        self.timeout = timeout

        # State
        self.board = chess.Board()

        # Specs
        specs = make_specs(self.device)
        self.done_spec = specs["done_spec"]
        self.reward_spec = specs["reward_spec"]
        self.observation_spec = specs["observation_spec"]
        self.state_spec = specs["state_spec"]
        self.action_spec = specs["action_spec"]
        self._state = specs["_state"]

        # Log done
        logger.debug(f"Created {self.__class__.__name__} env.")

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Step method.

        Args:
            tensordict (TensorDict):
                Input TensorDict, with `"action"` key.

        Returns:
            TensorDict: _description_
        """
        out = TensorDict(
            {
                "state": torch.zeros(8, 8).int().to(self.device),
                "reward": torch.Tensor([1]).float().to(self.device),
                "done": torch.Tensor([False]).bool().to(self.device),
            },
            batch_size=tensordict.shape,
            device=self.device,
        )
        return out

    def _reset(self, tensordict: TensorDict = None) -> TensorDict:
        """Reset."""
        if tensordict is None or tensordict.is_empty():
            batch_size = self.batch_size
        else:
            batch_size = tensordict.shape
        # Return
        out = TensorDict(
            {
                "state": torch.zeros(8, 8).int().to(self.device),
                "done": torch.Tensor([False]).bool().to(self.device),
            },
            batch_size=batch_size,
        )
        return out

    def _set_seed(self, seed: int) -> None:
        """The `_set_seed()` method sets the seed of any random number generator in the environment.

        Here we don't use any randomness but you can imagine a scenario where we initialize the state to a random value or add noise to the output observation in which case setting the random seed for reproducibility purposes would be very helpfull.

        Args:
            seed (int):
                Seed for RNG.
        """

    def sample(self, from_engine: bool = True) -> ty.Optional[TensorDict]:
        """Samples a legal action (chess move).

        Args:
            from_engine (bool):
                If `False`, a random legal movei is selected.

        Returns:
            (TensorDict): TensorDict with the sampled action.
        """
        if from_engine:
            # Get move from engine
            with SimpleEngine.popen_uci(self.engine_path) as engine:
                logger.trace(f"Sampling a move from {self.engine_path}")
                result = engine.play(self.board, chess.engine.Limit(time=self.timeout))
                move = result.move
        else:
            # Get random move
            move = get_random_move(self.board)

        # Early stop condition
        if move is None:  # pragma: no cover
            logger.warning("No legal move by engine...")
            return None

        # Get action tensor
        move_idx = action_dict()[move.uci()]
        action = torch.Tensor([move_idx]).to(self.action_spec.dtype).to(self.device)

        # Return TensorDict
        td = TensorDict(
            {"action": action},
            batch_size=torch.Size(),
            device=self.device,
        )
        return td
