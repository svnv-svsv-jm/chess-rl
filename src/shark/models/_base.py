__all__ = ["BaseRL"]

from loguru import logger
import typing as ty

import torch
from torchrl.envs import EnvBase, GymEnv

from .loops import RLTrainingLoop
from .utils import initialize, initialize_actor


class BaseRL(RLTrainingLoop):
    """Base for RL Model. See: https://pytorch.org/rl/tutorials/coding_ppo.html#training-loop"""

    def __init__(
        self,
        actor_nn: torch.nn.Module,
        value_nn: torch.nn.Module = None,
        env_name: str = "InvertedDoublePendulum-v4",
        model: str = "ppo",
        gamma: float = 0.99,
        lmbda: float = 0.95,
        entropy_eps: float = 1e-4,
        clip_epsilon: float = 0.2,
        alpha_init: float = 1,
        loss_function: str = "smooth_l1",
        flatten_state: bool = False,
        tau: float = 1e-2,
        discrete: bool = False,
        qvalue_actor: bool = False,
        **kwargs: ty.Any,
    ) -> None:
        """
        Args:
            env (ty.Union[str, EnvBase], optional): _description_. Defaults to "InvertedDoublePendulum-v4".
            num_cells (int, optional): _description_. Defaults to 256.
            lr (float, optional): _description_. Defaults to 3e-4.
            max_grad_norm (float, optional): _description_. Defaults to 1.0.
            frame_skip (int, optional): _description_. Defaults to 1.
            frames_per_batch (int, optional): _description_. Defaults to 100.
            total_frames (int, optional): _description_. Defaults to 100_000.
            accelerator (ty.Union[str, torch.device], optional): _description_. Defaults to "cpu".
            sub_batch_size (int, optional):
                Cardinality of the sub-samples gathered from the current data in the inner loop.
                Defaults to `1`.
            clip_epsilon (float, optional): _description_. Defaults to 0.2.
            gamma (float, optional): _description_. Defaults to 0.99.
            lmbda (float, optional): _description_. Defaults to 0.95.
            entropy_eps (float, optional): _description_. Defaults to 1e-4.
            lr_monitor (str, optional): _description_. Defaults to "loss/train".
            lr_monitor_strict (bool, optional): _description_. Defaults to False.
            rollout_max_steps (int, optional): _description_. Defaults to 1000.
            n_mlp_layers (int, optional): _description_. Defaults to 3.
            flatten (bool, optional): _description_. Defaults to False.
            flatten_start_dim (int, optional): _description_. Defaults to 0.
            legacy (bool, optional): _description_. Defaults to False.
            automatic_optimization (bool, optional): _description_. Defaults to True.
        """
        self.save_hyperparameters(
            ignore=[
                "base_env",
                "env",
                "loss_module",
                "policy_module",
                "value_module",
                "actor_nn",
                "value_nn",
            ]
        )
        self.discrete = discrete
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_eps = entropy_eps
        self.env_name = env_name
        self.device_info = kwargs.get("device", "cpu")
        self.frame_skip = kwargs.get("frame_skip", 1)
        # Environment
        base_env = self.make_env()
        # Env transformations
        env = self.transformed_env(base_env)
        # Specs
        observation_spec = base_env.observation_spec["observation"]
        # Sanity check
        logger.debug(f"observation_spec: {observation_spec}")
        logger.debug(f"reward_spec: {base_env.reward_spec}")
        logger.debug(f"done_spec: {base_env.done_spec}")
        logger.debug(f"action_spec: {base_env.action_spec}")
        logger.debug(f"state_spec: {base_env.state_spec}")
        # Actor
        policy_module = initialize_actor(
            actor_nn=actor_nn,
            env=env,
            flatten_state=flatten_state,
            qvalue=qvalue_actor,
        )
        logger.debug(f"Initialized policy: {policy_module}")
        # Critic and loss depend on the model
        modules = initialize(
            policy_module=policy_module,
            env=env,
            model=model,
            discrete=discrete,
            value_nn=value_nn,
            flatten_state=flatten_state,
            loss_function=loss_function,
            alpha_init=alpha_init,
            tau=tau,
            gamma=gamma,
            lmbda=lmbda,
            clip_epsilon=clip_epsilon,
            entropy_bonus=bool(self.entropy_eps),
            entropy_coef=self.entropy_eps,
        )
        loss_module = modules["loss_module"]
        advantage_module = modules["advantage_module"]
        target_net_updater = modules["target_net_updater"]
        # Call superclass
        super().__init__(
            loss_module=loss_module,
            policy_module=policy_module,
            advantage_module=advantage_module,
            target_net_updater=target_net_updater,
            **kwargs,
        )

    def make_env(self) -> EnvBase:
        """Utility function to init an env.

        Args:
            env (ty.Union[str, EnvBase]): _description_

        Returns:
            EnvBase: _description_
        """
        env = GymEnv(
            self.env_name,
            device=self.device_info,
            frame_skip=self.frame_skip,
        )
        return env
