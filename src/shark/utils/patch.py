__all__ = [
    "EnvBase",
    "ParallelEnv",
    "SerialEnv",
    "step_and_maybe_reset",
    "_cache_values",
    "transform_observation_spec",
]

from collections import OrderedDict
from loguru import logger
import typing as ty
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import (
    DTypeCastTransform,
    SerialEnv as SerialEnvRL,
    EnvBase as EnvBaseRL,
    ParallelEnv as ParallelEnvRL,
)
from torchrl.envs.utils import _terminated_or_truncated, step_mdp
from torchrl.data import TensorSpec


class SerialEnv(SerialEnvRL):
    """Patched SerialEnv..."""

    def step_and_maybe_reset(
        self,
        tensordict: TensorDictBase,
    ) -> ty.Tuple[TensorDictBase, TensorDictBase]:
        """Patched."""
        return step_and_maybe_reset(self, tensordict)


class EnvBase(EnvBaseRL):
    """Patched EnvBase..."""

    def step_and_maybe_reset(
        self,
        tensordict: TensorDictBase,
    ) -> ty.Tuple[TensorDictBase, TensorDictBase]:
        """Patched."""
        return step_and_maybe_reset(self, tensordict)


class ParallelEnv(ParallelEnvRL):
    """Patched ParallelEnv..."""

    def step_and_maybe_reset(
        self,
        tensordict: TensorDictBase,
    ) -> ty.Tuple[TensorDictBase, TensorDictBase]:
        """Patched."""
        return step_and_maybe_reset(self, tensordict)


def step_and_maybe_reset(
    self: EnvBaseRL,
    tensordict: TensorDictBase,
) -> ty.Tuple[TensorDictBase, TensorDictBase]:
    """Runs a step in the environment and (partially) resets it if needed.

    Args:
        tensordict (TensorDictBase): an input data structure for the :meth:`~.step`
            method.

    This method allows to easily code non-stopping rollout functions.

    Examples:
        >>> from torchrl.envs import ParallelEnv, GymEnv
        >>> def rollout(env, n):
        ...     data_ = env.reset()
        ...     result = []
        ...     for i in range(n):
        ...         data, data_ = env.step_and_maybe_reset(data_)
        ...         result.append(data)
        ...     return torch.stack(result)
        >>> env = ParallelEnv(2, lambda: GymEnv("CartPole-v1"))
        >>> print(rollout(env, 2))
        TensorDict(
            fields={
                done: Tensor(shape=torch.Size([2, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([2, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([2, 2, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([2, 2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([2, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([2, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([2, 2]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([2, 2, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([2, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([2, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([2, 2]),
            device=cpu,
            is_shared=False)
    """
    logger.trace(f"step_and_maybe_reset...")
    action: torch.Tensor = tensordict["action"]
    logger.trace(f"{action.size()}")
    assert action.size() == self.action_spec.shape, f"{self.action_spec.shape} but {action.size()}"
    tensordict = self.step(tensordict)
    # done and truncated are in done_keys
    # We read if any key is done.
    tensordict_ = step_mdp(
        tensordict,
        keep_other=True,
        exclude_action=False,
        exclude_reward=False,  # NOTE: Patched this
        reward_keys=self.reward_keys,
        action_keys=self.action_keys,
        done_keys=self.done_keys,
    )
    any_done = _terminated_or_truncated(
        tensordict_,
        full_done_spec=self.output_spec["full_done_spec"],
        key="_reset",
    )
    if any_done:
        tensordict_ = self.reset(tensordict_)
    # if isinstance(tensordict, (TensorDict, TensorDictBase)):
    #     assert "reward" in tensordict.keys(), f"{tensordict}"
    # if isinstance(tensordict_, (TensorDict, TensorDictBase)):
    #     assert "reward" in tensordict_.keys(), f"{tensordict_}"
    logger.trace(f"tensordict: {tensordict}")
    logger.trace(f"tensordict_: {tensordict_}")
    return tensordict, tensordict_


def _cache_values(fun: ty.Callable) -> ty.Callable:
    """Caches the tensordict returned by a property."""
    name = fun.__name__

    def new_fun(self: ty.Any, netname: str = None) -> ty.Any:
        __dict__: dict = self.__dict__
        __dict__.setdefault("_cache", {})
        _cache = __dict__["_cache"]
        attr_name = name
        if netname is not None:
            attr_name += "_" + netname
        if attr_name in _cache:
            out = _cache[attr_name]
            return out
        if netname is not None:
            out = fun(self, netname)
        else:
            out = fun(self)
        # TODO: decide what to do with locked tds in functional calls
        # if is_tensor_collection(out):
        #     out.lock_()
        _cache[attr_name] = out
        return out

    return new_fun


def transform_observation_spec(  # pragma: no cover
    self: DTypeCastTransform,
    observation_spec: TensorSpec,
) -> TensorSpec:
    """No idea."""
    full_observation_spec = observation_spec
    for observation_key, observation_spec in list(full_observation_spec.items(True, True)):
        # find out_key that match the in_key
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            if observation_key == in_key:
                if observation_spec.dtype != self.dtype_in:
                    self.dtype_in = observation_spec.dtype
                    # raise TypeError(
                    #     f"observation_spec.dtype is not {self.dtype_in}"
                    # )
                full_observation_spec[out_key] = self._transform_spec(observation_spec)
    return full_observation_spec
