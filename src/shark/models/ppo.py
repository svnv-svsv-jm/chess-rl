__all__ = ["PPO"]

from loguru import logger
import typing as ty

from collections import defaultdict
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from lightning.fabric.utilities.types import LRScheduler
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig, LRSchedulerConfigType
from lightning.pytorch.core.optimizer import LightningOptimizer
import torch
from torch import nn, Tensor
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
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from shark.utils import find_device


class PPO(pl.LightningModule):
    """Basic PPO Model. See: https://pytorch.org/rl/tutorials/coding_ppo.html#training-loop"""

    def __init__(
        self,
        env_name: str = "InvertedDoublePendulum-v4",
        num_cells: int = 256,
        lr: float = 3e-4,
        max_grad_norm: float = 1.0,
        frame_skip: int = 1,
        frames_per_batch: int = 100,
        total_frames: int = None,
        accelerator: ty.Union[str, torch.device] = "cpu",
        sub_batch_size: int = 64,  # cardinality of the sub-samples gathered from the current data in the inner loop
        num_epochs: int = 10,  # optimisation steps per batch of data collected
        # clip value for PPO loss: see the equation in the intro for more context.
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        lmbda: float = 0.95,
        entropy_eps: float = 1e-4,
        lr_monitor: str = "loss/train",
        lr_monitor_strict: bool = False,
        rollout_max_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.max_grad_norm = max_grad_norm
        self.num_cells = num_cells
        self.lr = lr
        self.frame_skip = frame_skip
        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames
        self.sub_batch_size = sub_batch_size
        self.num_epochs = num_epochs
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_eps = entropy_eps
        self.rollout_max_steps = rollout_max_steps
        device = find_device(accelerator)
        self.base_env = GymEnv(
            env_name,
            device=device,
            frame_skip=frame_skip,
        )
        obs_norm = ObservationNorm(in_keys=["observation"])
        self.env = TransformedEnv(
            self.base_env,
            Compose(
                # normalize observations
                obs_norm,
                DoubleToFloat(
                    in_keys=["observation"],
                ),
                StepCounter(),
            ),
        )
        obs_norm.init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
        logger.debug(f"normalization constant shape: {self.env.transform[0].loc.shape}")
        logger.debug(f"observation_spec: {self.env.observation_spec}")
        logger.debug(f"reward_spec: {self.env.reward_spec}")
        logger.debug(f"done_spec: {self.env.done_spec}")
        logger.debug(f"action_spec: {self.env.action_spec}")
        logger.debug(f"state_spec: {self.env.state_spec}")
        check_env_specs(self.env)
        # Actor
        self.actor_net = nn.Sequential(
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(2 * self.env.action_spec.shape[-1], device=device),
            NormalParamExtractor(),
        )
        self.policy_module = TensorDictModule(
            self.actor_net,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        )
        self.policy_module = ProbabilisticActor(
            module=self.policy_module,
            spec=self.env.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "min": self.env.action_spec.space.minimum,
                "max": self.env.action_spec.space.maximum,
            },
            return_log_prob=True,
            # we'll need the log-prob for the numerator of the importance weights
        )
        # Critic
        self.value_net = nn.Sequential(
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(1, device=device),
        )
        self.value_module = ValueOperator(
            module=self.value_net,
            in_keys=["observation"],
        )
        # ReplayBuffer
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(frames_per_batch),
            sampler=SamplerWithoutReplacement(),
        )
        # Loss
        self.advantage_module = GAE(
            gamma=gamma,
            lmbda=lmbda,
            value_network=self.value_module,
            average_gae=True,
        )
        self.loss_module = ClipPPOLoss(
            actor=self.policy_module,
            critic=self.value_module,
            clip_epsilon=clip_epsilon,
            entropy_bonus=bool(entropy_eps),
            entropy_coef=entropy_eps,
            # these keys match by default but we set this for completeness
            value_target_key=self.advantage_module.value_target_key,
            critic_coef=1.0,
            gamma=0.99,
            loss_critic_type="smooth_l1",
        )
        # Important: This property activates manual optimization
        self.automatic_optimization = False
        # Optim attr: will exist only after training initialisation
        self.optimizer: torch.optim.Adam
        self.scheduler: torch.optim.lr_scheduler.CosineAnnealingLR
        self.lr_monitor = lr_monitor
        self.lr_monitor_strict = lr_monitor_strict

    def get_collector(self) -> SyncDataCollector:
        """Create `SyncDataCollector`."""
        collector = SyncDataCollector(
            self.env,
            self.policy_module,
            frames_per_batch=self.frames_per_batch,
            total_frames=self.total_frames,
            split_trajs=False,
            device=self.device,
        )
        return collector

    def train_dataloader(self) -> SyncDataCollector:
        """Create DataLoader for training."""
        collector = self.get_collector()
        return collector

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configures the optimizer (`torch.optim.Adam`) and the learning rate scheduler (`torch.optim.lr_scheduler.CosineAnnealingLR`)."""
        self.optimizer = torch.optim.Adam(self.loss_module.parameters(), self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            max(1, self.trainer.max_steps // self.frames_per_batch),
            0.0,
        )
        lr_scheduler = LRSchedulerConfigType(  # type: ignore
            scheduler=self.scheduler,
            monitor=self.lr_monitor,
            strict=self.lr_monitor_strict,
        )
        cfg = OptimizerLRSchedulerConfig(optimizer=self.optimizer, lr_scheduler=lr_scheduler)
        return cfg

    def training_step(self, batch: TensorDict, batch_idx: int) -> Tensor:
        """Implementation follows the PyTorch tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html"""
        optimizer = self.optimizers()
        assert isinstance(optimizer, (torch.optim.Optimizer, LightningOptimizer))
        # Run optimization step
        loss = self.step(batch, batch_idx=batch_idx, tag="train")
        # Zero grad before accumulating them
        optimizer.zero_grad()
        # Run backward
        logger.trace("Running manual_backward()")
        self.manual_backward(loss)
        # Clip gradients if necessary
        clip_val = self.trainer.gradient_clip_val
        if clip_val is None:
            clip_val = self.max_grad_norm
        logger.trace(f"Clipping gradients to {clip_val}")
        torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), clip_val)
        # Optimizer
        logger.trace("Running optimizer.step()")
        optimizer.step()
        # Call schedulers
        self.call_scheduler()
        # We evaluate the policy once every `sefl.trainer.val_check_interval` batches of data
        n = self.trainer.val_check_interval
        if n is None:
            n = 10
        n = int(n)
        if batch_idx % n == 0:
            self.rollout()
        return loss

    def call_scheduler(self) -> None:
        """Call schedulers. We are using an infinite datalaoder, this will never be called by the `pl.Trainer` in the `on_train_epoch_end` hook. We have to call it manually in the `training_step`."""
        scheduler = self.lr_schedulers()
        assert isinstance(scheduler, LRScheduler)
        try:
            # c = self.trainer.callback_metrics[self.lr_monitor]
            scheduler.step(self.trainer.global_step)
        except Exception as ex:
            logger.warning(ex)

    def step(
        self,
        tensordict_data: TensorDict,
        batch_idx: int = None,
        tag: str = "train",
    ) -> Tensor:
        """Common step."""
        logger.trace(f"[{batch_idx}] Batch: {tensordict_data.batch_size}")
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        with torch.no_grad():
            self.advantage_module(tensordict_data)
        data_view: TensorDict = tensordict_data.reshape(-1)
        self.replay_buffer.extend(data_view.cpu())
        loss = torch.tensor(0.0).to(self.device)
        for _ in range(self.frames_per_batch // self.sub_batch_size):
            subdata: TensorDict = self.replay_buffer.sample(self.sub_batch_size)
            loss_vals = self.loss_module(subdata.to(self.device))
            loss_objective: Tensor = loss_vals["loss_objective"]
            loss_critic: Tensor = loss_vals["loss_critic"]
            loss_entropy: Tensor = loss_vals["loss_entropy"]
            loss_value = loss_objective + loss_critic + loss_entropy
            loss += loss_value
            # # Optimization: backward, grad clipping and optim step
            # loss_value.backward()
            # # this is not strictly mandatory but it's good practice to keep
            # # your gradient norm bounded
            # torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), self.trainer.gradient_clip_val)
            # optim.step()
            # optim.zero_grad()
        self.log(f"loss/{tag}", loss, prog_bar=True)
        reward: Tensor = tensordict_data["next", "reward"]
        self.log(f"reward/{tag}", reward.mean().item(), prog_bar=True)
        step_count: Tensor = tensordict_data["step_count"]
        self.log(f"step_count/{tag}", step_count.max().item(), prog_bar=True)
        return loss

    def rollout(self, tag: str = "eval") -> None:
        """We evaluate the policy once every `sefl.trainer.val_check_interval` batches of data.
        Evaluation is rather simple: execute the policy without exploration (take the expected value of the action distribution) for a given number of steps.
        The `self.env.rollout()` method can take a policy as argument: it will then execute this policy at each step.
        """
        logger.trace("Rollout...")
        with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = self.env.rollout(self.rollout_max_steps, self.policy_module)
            reward = eval_rollout["next", "reward"]
            self.log(f"reward/{tag}", reward.mean().item())
            self.log(f"reward_sum/{tag}", reward.sum().item())
            step_count = eval_rollout["step_count"]
            self.log(f"step_count/{tag}", step_count.max().item())
            del eval_rollout
