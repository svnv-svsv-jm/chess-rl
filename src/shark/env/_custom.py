__all__ = ["_CustomEnv"]

from loguru import logger
import typing as ty

import torch

from tensordict import TensorDict
from torchrl.data import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    BinaryDiscreteTensorSpec,
    DiscreteTensorSpec,
)
from torchrl.envs import EnvBase


class _CustomEnv(EnvBase):
    """Custom dummy environment."""

    def __init__(
        self,
        **kwargs: ty.Any,
    ) -> None:
        super().__init__(**kwargs)  # call the constructor of the base class
        # Action is a one-hot tensor
        self.action_spec = DiscreteTensorSpec(
            n=10,
            shape=(),
            device=self.device,
            dtype=torch.float32,
        )
        # Observation space
        observation_spec = DiscreteTensorSpec(
            n=13,
            shape=(8, 8),
            device=self.device,
            dtype=torch.float32,
        )
        self.observation_spec = CompositeSpec(observation=observation_spec)
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
        # Return new TensorDict
        return TensorDict(
            {
                "observation": torch.zeros(
                    (8, 8, 13), dtype=self.observation_spec.dtype, device=self.device
                ),
                "reward": torch.Tensor([0]).to(self.reward_spec.dtype).to(self.device),
                "done": False,
            },
            batch_size=torch.Size(),
            device=self.device,
        )

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """The `_step()` method takes in a `TensorDict` from which it reads an action, applies the action and returns a new `TensorDict` containing the observation, reward and done signal for that timestep.

        Args:
            tensordict (TensorDict): _description_

        Returns:
            TensorDict: _description_
        """
        # Return new TensorDict
        td = TensorDict(
            {
                "observation": torch.zeros(
                    (8, 8, 13), dtype=self.observation_spec.dtype, device=self.device
                ),
                "reward": torch.Tensor([0]).to(self.reward_spec.dtype).to(self.device),
                "done": True,
            },
            batch_size=torch.Size(),
            device=self.device,
        )
        logger.trace(f"Returning new TensorDict: {td}")
        return td

    def _set_seed(self, seed: int) -> None:
        """The `_set_seed()` method sets the seed of any random number generator in the environment.

        Here we don't use any randomness but you can imagine a scenario where we initialize the state to a random value or add noise to the output observation in which case setting the random seed for reproducibility purposes would be very helpfull.

        Args:
            seed (int):
                Seed for RNG.
        """
