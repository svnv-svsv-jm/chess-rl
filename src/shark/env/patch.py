__all__ = ["step_and_maybe_reset"]

from loguru import logger
import typing as ty
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import EnvBase
from torchrl.envs.utils import _terminated_or_truncated, step_mdp


def step_and_maybe_reset(
    self: EnvBase,
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
    tensordict = self.step(tensordict)
    if isinstance(tensordict, (TensorDict, TensorDictBase)):
        assert "reward" in tensordict.keys()
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
    if isinstance(tensordict_, (TensorDict, TensorDictBase)):
        assert "reward" in tensordict_.keys(), f"{tensordict_}"
    any_done = _terminated_or_truncated(
        tensordict_,
        full_done_spec=self.output_spec["full_done_spec"],
        key="_reset",
    )
    if any_done:
        tensordict_ = self.reset(tensordict_)
    if isinstance(tensordict, (TensorDict, TensorDictBase)):
        assert "reward" in tensordict.keys(), f"{tensordict}"
    if isinstance(tensordict_, (TensorDict, TensorDictBase)):
        assert "reward" in tensordict_.keys(), f"{tensordict_}"
    logger.trace(f"tensordict: {tensordict}")
    logger.trace(f"tensordict_: {tensordict_}")
    return tensordict, tensordict_
