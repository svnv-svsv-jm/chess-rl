__all__ = ["PPO", "PPOChess"]

from loguru import logger
import typing as ty

import lightning.pytorch as pl
from lightning.fabric.utilities.types import LRScheduler
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig, LRSchedulerConfigType
from lightning.pytorch.core.optimizer import LightningOptimizer
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
    FlattenObservation,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, MLP
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from shark.datasets import CollectorDataset
from shark.utils import find_device
from ._base import BaseRL


class PPO(BaseRL):
    """Basic PPO Model. See: https://pytorch.org/rl/tutorials/coding_ppo.html#training-loop"""

    def __init__(
        self,
        env: ty.Union[str, EnvBase] = "InvertedDoublePendulum-v4",
        num_cells: int = 256,
        gamma: float = 0.99,
        lmbda: float = 0.95,
        entropy_eps: float = 1e-4,
        clip_epsilon: float = 0.2,
        n_mlp_layers: int = 3,
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
            in_keys (ty.List[str], optional): _description_. Defaults to ["observation"].
            flatten (bool, optional): _description_. Defaults to False.
            flatten_start_dim (int, optional): _description_. Defaults to 0.
            legacy (bool, optional): _description_. Defaults to False.
            automatic_optimization (bool, optional): _description_. Defaults to True.
        """
        self.save_hyperparameters(ignore=["env"])
        self.num_cells = num_cells
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_eps = entropy_eps
        # Environment
        if isinstance(env, str):
            self.base_env = GymEnv(
                env,
                device=kwargs.get("device", None),
                frame_skip=kwargs.get("frame_skip", None),
            )
        else:
            self.base_env = env
        # Env transformations
        self.env = self.transformed_env(self.base_env)
        # Sanity check
        logger.debug(f"observation_spec: {self.env.observation_spec}")
        logger.debug(f"reward_spec: {self.env.reward_spec}")
        logger.debug(f"done_spec: {self.env.done_spec}")
        logger.debug(f"action_spec: {self.env.action_spec}")
        logger.debug(f"state_spec: {self.env.state_spec}")
        # Actor
        shape = self.env.observation_spec["observation"].shape
        assert isinstance(shape, torch.Size)
        out_features = self.env.action_spec.shape[-1]
        logger.debug(f"MLP out_shape: {out_features}")
        self.actor_net = torch.nn.Sequential(
            MLP(
                out_features=2 * out_features,
                depth=n_mlp_layers,
                num_cells=num_cells,
                dropout=True,
            ),
            NormalParamExtractor(),
        )
        logger.debug(f"Initialized actor: {self.actor_net}")
        policy_module = TensorDictModule(
            self.actor_net,
            in_keys=self.in_keys,
            out_keys=["loc", "scale"],
        )
        td = self.env.reset()
        policy_module(td)
        self.policy_module = ProbabilisticActor(
            module=policy_module,
            spec=self.env.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "min": 0,  # self.env.action_spec.space.minimum,
                "max": 1,  # self.env.action_spec.space.maximum,
            },
            return_log_prob=True,  # we'll need the log-prob for the numerator of the importance weights
        )
        logger.debug(f"Initialized policy: {self.policy_module}")
        # Critic
        self.value_net = MLP(
            out_features=1,
            depth=n_mlp_layers,
            num_cells=num_cells,
            dropout=True,
        )
        logger.debug(f"Initialized critic: {self.value_net}")
        value_module = ValueOperator(
            module=self.value_net,
            in_keys=self.in_keys,
        )
        td = self.env.reset()
        value_module(td)
        # Loss
        advantage_module = GAE(
            gamma=gamma,
            lmbda=lmbda,
            value_network=self.value_module,
            average_gae=True,
        )
        loss_module = ClipPPOLoss(
            actor=self.policy_module,
            critic=self.value_module,
            clip_epsilon=clip_epsilon,
            entropy_bonus=bool(entropy_eps),
            entropy_coef=entropy_eps,
            # these keys match by default but we set this for completeness
            critic_coef=1.0,
            # gamma=0.99,
            loss_critic_type="smooth_l1",
        )
        loss_module.set_keys(value_target=self.advantage_module.value_target_key)
        # Call superclass
        super().__init__(
            env=env,
            loss_module=loss_module,
            policy_module=policy_module,
            value_module=value_module,
            advantage_module=advantage_module,
            **kwargs,
        )

    def transformed_env(self, base_env: EnvBase) -> EnvBase:
        """Setup transformed environment."""
        obs_norm = ObservationNorm(in_keys=self.in_keys)
        env = TransformedEnv(
            base_env,
            transform=Compose(
                obs_norm,
                DoubleToFloat(in_keys=self.in_keys),
                StepCounter(),
            ),
        )
        obs_norm.init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
        logger.debug(f"normalization constant shape: {env.transform[0].loc.shape}")
        return env


class PPOChess(PPO):
    """Same but overrides the `transformed_env` method."""

    def transformed_env(self, base_env: EnvBase) -> EnvBase:
        """Setup transformed environment."""
        env = TransformedEnv(
            base_env,
            transform=Compose(
                StepCounter(),
                # FlattenObservation(
                #     first_dim=1,
                #     last_dim=-1,
                #     in_keys=self.in_keys,
                #     allow_positive_dim=True,
                # ),
            ),
        )
        return env
