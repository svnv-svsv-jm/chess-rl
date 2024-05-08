__all__ = ["SyncDataCollector"]

import typing as ty
from loguru import logger

import torch
from tensordict import TensorDictBase, TensorDict
from torchrl.envs.utils import set_exploration_type
from torchrl.collectors import (
    SyncDataCollector as SyncDataCollector_,
    MultiSyncDataCollector,
    MultiaSyncDataCollector,
)


class SyncDataCollector(SyncDataCollector_):
    """Patched `SyncDataCollector`."""

    @torch.no_grad()
    def rollout(self) -> TensorDictBase:
        """Computes a rollout in the environment using the provided policy.

        Returns:
            TensorDictBase containing the computed rollout.
        """
        # Help mypy
        self._shuttle: TensorDictBase
        self._final_rollout: TensorDictBase
        if not isinstance(self._shuttle, TensorDictBase):
            raise RuntimeError(f"self._shuttle not {TensorDictBase} but {type(self._shuttle)}")
        if not isinstance(self._final_rollout, TensorDictBase):
            raise RuntimeError(
                f"self._shuttle not {TensorDictBase} but {type(self._final_rollout)}"
            )

        if self.reset_at_each_iter:
            self._shuttle.update(self.env.reset())

        self._final_rollout.fill_(("collector", "traj_ids"), -1)
        logger.trace(f"_final_rollout: {self._final_rollout}")
        tensordicts: ty.List[TensorDict] = []
        with set_exploration_type(self.exploration_type):
            for t in range(self.frames_per_batch):
                if self.init_random_frames is not None and self._frames < self.init_random_frames:
                    self.env.rand_action(self._shuttle)
                else:
                    if self._cast_to_policy_device:
                        if self.policy_device is not None:
                            policy_input = self._shuttle.to(self.policy_device, non_blocking=True)
                            self._sync_policy()
                        elif self.policy_device is None:
                            # we know the tensordict has a device otherwise we would not be here
                            # we can pass this, clear_device_ must have been called earlier
                            # policy_input = self._shuttle.clear_device_()
                            policy_input = self._shuttle
                    else:
                        policy_input = self._shuttle
                    # we still do the assignment for security
                    policy_output = self.policy(policy_input)
                    if self._shuttle is not policy_output:
                        # ad-hoc update shuttle
                        self._shuttle.update(policy_output, keys_to_update=self._policy_output_keys)

                if self._cast_to_env_device:
                    if self.env_device is not None:
                        env_input = self._shuttle.to(self.env_device, non_blocking=True)
                        self._sync_env()
                    elif self.env_device is None:
                        # we know the tensordict has a device otherwise we would not be here
                        # we can pass this, clear_device_ must have been called earlier
                        # env_input = self._shuttle.clear_device_()
                        env_input = self._shuttle
                else:
                    env_input = self._shuttle
                env_output, env_next_output = self.env.step_and_maybe_reset(env_input)

                if self._shuttle is not env_output:
                    # ad-hoc update shuttle
                    next_data = env_output.get("next")
                    if self._shuttle_has_no_device:
                        # Make sure
                        next_data.clear_device_()
                    self._shuttle.set("next", next_data)

                if self.storing_device is not None:
                    t = self._shuttle.to(self.storing_device, non_blocking=True)
                    logger.trace(f"Appending {t}")
                    tensordicts.append(t)
                    self._sync_storage()
                else:
                    logger.trace(f"Appending {self._shuttle}")
                    tensordicts.append(self._shuttle)

                # carry over collector data without messing up devices
                collector_data = self._shuttle.get("collector").copy()
                self._shuttle = env_next_output
                if self._shuttle_has_no_device:
                    self._shuttle.clear_device_()
                self._shuttle.set("collector", collector_data)

                self._update_traj_ids(env_output)

                if self.interruptor is not None and self.interruptor.collection_stopped():
                    try:
                        torch.stack(
                            tensordicts,
                            self._final_rollout.ndim - 1,
                            out=self._final_rollout[..., : t + 1],
                        )
                    except RuntimeError:
                        with self._final_rollout.unlock_():
                            torch.stack(
                                tensordicts,
                                self._final_rollout.ndim - 1,
                                out=self._final_rollout[..., : t + 1],
                            )
                    break
            else:
                try:
                    logger.trace(f"_final_rollout: {self._final_rollout}")
                    logger.trace(f"tensordicts: {tensordicts}")
                    # PATCH BEGIN
                    key = "action"
                    a: torch.Tensor = self._final_rollout[key]
                    for t in tensordicts:
                        assert isinstance(t, TensorDict)
                        if key in t.keys():
                            dtype = t[key].dtype
                            self._final_rollout[key] = a.to(dtype)
                            break
                    else:
                        raise RuntimeError(f"Could not find {key} in {tensordicts}")
                    # PATCH END
                    self._final_rollout = torch.stack(
                        tensordicts,
                        self._final_rollout.ndim - 1,
                        out=self._final_rollout,
                    )
                except RuntimeError:
                    with self._final_rollout.unlock_():
                        self._final_rollout = torch.stack(
                            tensordicts,
                            self._final_rollout.ndim - 1,
                            out=self._final_rollout,
                        )
        return self._maybe_set_truncated(self._final_rollout)
