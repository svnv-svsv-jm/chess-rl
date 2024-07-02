__all__ = ["Chess"]

from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn

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

from .const import N_PIECES, N_ACTIONS


class Chess(EnvBase):
    """Chess RL environment."""

    def __init__(
        self,
        *,
        device: torch.device | str | int | None = None,
        batch_size: torch.Size | None = None,
        run_type_checks: bool = True,
        allow_done_after_reset: bool = False
    ):
        """
        Args:
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

        # action_spec (TensorSpec): the spec of the action. Links to the spec of the leaf action if only one action tensor is to be expected
        self.action_spec = DiscreteTensorSpec(
            n=N_ACTIONS,
            shape=torch.Size([1]),
            device=self.device,
            dtype=torch.int,
        )

        # observation_spec (CompositeSpec): a composite spec such that `full_observation_spec.zero()` returns a tensordict containing only the leaves encoding the observation of the environment.
        # Observation space
        self._state = DiscreteTensorSpec(
            n=N_PIECES,
            shape=torch.Size([8, 8]),
            device=self.device,
            dtype=torch.int,
        )
        self.observation_spec = CompositeSpec(state=self._state)
        # since the environment is stateless, we expect the previous output as input.
        # For this, ``EnvBase`` expects some state_spec to be available
        self.state_spec = self.observation_spec.clone()

        # Unlimited reward space
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=torch.Size([1]),
            device=self.device,
            dtype=torch.float32,
        )

        # done_spec (CompositeSpec): equivalent to `full_done_spec` as all `done_specs` contain at least a `"done"` and a `"terminated"` entry
        self.done_spec = BinaryDiscreteTensorSpec(
            n=1,
            shape=torch.Size([1]),
            device=self.device,
            dtype=torch.bool,
        )

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
