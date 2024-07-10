__all__ = ["PlayingMode", "make_specs"]

import typing as ty
from loguru import logger

import torch
from torchrl.data import (
    TensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    BinaryDiscreteTensorSpec,
    DiscreteTensorSpec,
)

from svchess.utils.const import N_PIECES, N_ACTIONS


class PlayingMode:
    """Playing mode."""

    ENGINE: str = "engine"
    RANDOM: str = "random"
    HYBRID: str = "hybrid"

    @classmethod
    def all(cls) -> ty.List[str]:
        """Returns all possibilities."""
        return [cls.ENGINE.lower(), cls.RANDOM.lower(), cls.HYBRID.lower()]

    @classmethod
    def isin(cls, val: str) -> bool:
        """Whether value exists."""
        return val.lower() in cls.all()


def make_specs(device: torch.device) -> ty.Dict[str, TensorSpec]:
    """Helper to create specs."""
    # action_spec (TensorSpec): the spec of the action. Links to the spec of the leaf action if only one action tensor is to be expected
    action_spec = DiscreteTensorSpec(
        n=N_ACTIONS,
        shape=torch.Size([1]),
        device=device,
        dtype=torch.int,
    )
    logger.debug(f"Created action spec:\n{action_spec}")

    # observation_spec (CompositeSpec): a composite spec such that `full_observation_spec.zero()` returns a tensordict containing only the leaves encoding the observation of the environment.
    # Observation space
    _state = DiscreteTensorSpec(
        n=N_PIECES,
        shape=torch.Size([8, 8]),
        device=device,
        dtype=torch.int,
    )
    observation_spec = CompositeSpec(state=_state)
    # since the environment is stateless, we expect the previous output as input.
    # For this, ``EnvBase`` expects some state_spec to be available
    state_spec = observation_spec.clone()
    logger.debug(f"Created state spec:\n{observation_spec}")

    # Unlimited reward space
    reward_spec = UnboundedContinuousTensorSpec(
        shape=torch.Size([1]),
        device=device,
        dtype=torch.float32,
    )
    logger.debug(f"Created reward spec:\n{reward_spec}")

    # done_spec (CompositeSpec): equivalent to `full_done_spec` as all `done_specs` contain at least a `"done"` and a `"terminated"` entry
    done_spec = BinaryDiscreteTensorSpec(
        n=1,
        shape=torch.Size([1]),
        device=device,
        dtype=torch.bool,
    )
    logger.debug(f"Created done spec:\n{done_spec}")

    # Return
    return dict(
        done_spec=done_spec,
        reward_spec=reward_spec,
        observation_spec=observation_spec,
        state_spec=state_spec,
        action_spec=action_spec,
        _state=_state,
    )
