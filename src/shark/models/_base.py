__all__ = ["BaseRL"]

from loguru import logger
import typing as ty

import lightning.pytorch as pl
from lightning.fabric.utilities.types import LRScheduler
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig, LRSchedulerConfigType
from lightning.pytorch.core.optimizer import LightningOptimizer
import torch
from torch import Tensor
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type

from shark.datasets import CollectorDataset
from shark.utils.patch import step_and_maybe_reset


class BaseRL(pl.LightningModule):
    """Base RL model. See: https://pytorch.org/rl/tutorials/coding_ppo.html#training-loop"""

    def __init__(
        self,
        env: ty.Union[str, EnvBase],
        loss_module: TensorDictModule,
        policy_module: TensorDictModule,
        value_module: TensorDictModule,
        advantage_module: TensorDictModule,
        in_keys: ty.List[str],
        lr: float = 3e-4,
        max_grad_norm: float = 1.0,
        frame_skip: int = 1,
        frames_per_batch: int = 100,
        total_frames: int = 100_000,
        sub_batch_size: int = 1,
        lr_monitor: str = "loss/train",
        lr_monitor_strict: bool = False,
        rollout_max_steps: int = 1000,
        automatic_optimization: bool = True,
    ) -> None:
        """_summary_

        Args:
            env (ty.Union[str, EnvBase]): _description_
            loss_module (TensorDictModule): _description_
            policy_module (TensorDictModule): _description_
            value_module (TensorDictModule): _description_
            lr (float, optional): _description_. Defaults to 3e-4.
            max_grad_norm (float, optional): _description_. Defaults to 1.0.
            frame_skip (int, optional): _description_. Defaults to 1.
            frames_per_batch (int, optional): _description_. Defaults to 100.
            total_frames (int, optional): _description_. Defaults to 100_000.
            accelerator (ty.Union[str, torch.device], optional): _description_. Defaults to "cpu".
            sub_batch_size (int, optional): _description_. Defaults to 1.
            lr_monitor (str, optional): _description_. Defaults to "loss/train".
            lr_monitor_strict (bool, optional): _description_. Defaults to False.
            rollout_max_steps (int, optional): _description_. Defaults to 1000.
            in_keys (ty.List[str], optional): _description_. Defaults to ["observation"].
            legacy (bool, optional): _description_. Defaults to False.
            automatic_optimization (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "base_env",
                "env",
                "loss_module",
                "policy_module",
                "value_module",
            ]
        )
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.frame_skip = frame_skip
        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames
        self.sub_batch_size = sub_batch_size
        self.rollout_max_steps = rollout_max_steps
        self.in_keys = in_keys
        # Environment
        self.base_env = self._init_env(env, frame_skip=frame_skip)
        # Env transformations
        self.env = self.transformed_env(self.base_env)
        # Patch this method with your function
        self.env.step_and_maybe_reset = lambda arg: step_and_maybe_reset(self.env, arg)
        # Sanity check
        logger.debug(f"Env: {self.base_env}")
        logger.debug(f"observation_spec: {self.env.observation_spec}")
        logger.debug(f"reward_spec: {self.env.reward_spec}")
        logger.debug(f"done_spec: {self.env.done_spec}")
        logger.debug(f"action_spec: {self.env.action_spec}")
        logger.debug(f"state_spec: {self.env.state_spec}")
        # Modules
        self.loss_module = loss_module
        self.policy_module = policy_module
        self.value_module = value_module
        self.advantage_module = advantage_module
        # Important: This property activates manual optimization
        self.automatic_optimization = automatic_optimization
        # Will exist only after training initialisation
        self.optimizer: torch.optim.Adam
        self.scheduler: torch.optim.lr_scheduler.CosineAnnealingLR
        self.lr_monitor = lr_monitor
        self.lr_monitor_strict = lr_monitor_strict
        self._dataset: CollectorDataset

    @property
    def replay_buffer(self) -> ReplayBuffer:
        """Gets replay buffer from collector."""
        return self.dataset.replay_buffer

    @property
    def dataset(self) -> CollectorDataset:
        """Gets dataset."""
        if not hasattr(self, "_dataset") or not isinstance(
            getattr(self, "_dataset"), CollectorDataset
        ):
            self._dataset = CollectorDataset(
                env=self.env,
                policy_module=self.policy_module,
                frames_per_batch=self.frames_per_batch,
                total_frames=self.total_frames,
                device=self.device,
                # batch_size=self.sub_batch_size,
            )
        return self._dataset

    def setup(self, stage: str = None) -> None:
        """Set up collector."""
        logger.debug(f"device: {self.device}")

    def train_dataloader(self) -> ty.Iterable[TensorDict]:
        """Create DataLoader for training."""
        return self.dataset

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configures the optimizer (`torch.optim.Adam`) and the learning rate scheduler (`torch.optim.lr_scheduler.CosineAnnealingLR`)."""
        self.optimizer = torch.optim.Adam(self.loss_module.parameters(), self.lr)
        try:
            max_steps = self.trainer.max_steps
        except RuntimeError:
            max_steps = 1
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            max(1, max_steps // self.frames_per_batch),
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
        # Run optimization step
        loss = self.step(batch, batch_idx=batch_idx, tag="train")
        # This will run only if manual optimization
        self.manual_optimization_step(loss)
        # We evaluate the policy once every `sefl.trainer.val_check_interval` batches of data
        n = self.trainer.val_check_interval
        if n is None:
            n = 10
        n = int(n)
        if batch_idx % n == 0:
            self.rollout()
        # Return loss
        return loss

    def on_train_epoch_end(self) -> None:
        """Check if we have to stop. For some reason, Lightning can't understand this. Probably because we are using an `IterableDataset`."""
        # Stop on max steps
        global_step = self.trainer.global_step
        max_steps = self.trainer.max_steps
        if global_step > max_steps:
            self.stop("global_step > max_steps")
            return
        # Stop on max epochs
        current_epoch = self.trainer.current_epoch
        max_epochs = self.trainer.max_epochs
        if isinstance(max_epochs, int) and current_epoch > max_epochs:
            self.stop("current_epoch > max_epochs")
            return
        # Stop on total frames
        if global_step > self.total_frames:
            self.stop("global_step > total_frames")
            return

    def stop(self, msg: str = "") -> None:
        """Change `Trainer` flat to make this stop."""
        self.trainer.should_stop = True
        logger.debug(f"Stopping. {msg}")

    def manual_optimization_step(self, loss: Tensor) -> None:
        """Steps to run if manual optimization is enabled."""
        if self.automatic_optimization:
            logger.trace("Automatic optimization is enabled, skipping manual optimization step.")
            return
        # Get optimizers
        optimizer = self.optimizers()
        assert isinstance(optimizer, (torch.optim.Optimizer, LightningOptimizer))
        # Zero grad before accumulating them
        optimizer.zero_grad()
        # Run backward
        logger.trace("Running manual_backward()")
        self.manual_backward(loss)
        # Clip gradients if necessary
        self.clip_gradients()
        # Optimizer
        logger.trace("Running optimizer.step()")
        optimizer.step()
        # Call schedulers
        self.call_scheduler()

    def clip_gradients(
        self,
        optimizer: torch.optim.Optimizer = None,
        gradient_clip_val: ty.Union[int, float] = None,
        gradient_clip_algorithm: str = None,
    ) -> None:
        """Clip gradients if necessary. This is an official hook."""
        clip_val = self.trainer.gradient_clip_val
        if clip_val is None:
            clip_val = self.max_grad_norm
        logger.trace(f"Clipping gradients to {clip_val}")
        torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), clip_val)

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
        batch: TensorDict,
        batch_idx: int = None,
        tag: str = "train",
    ) -> Tensor:
        """Common step."""
        logger.trace(f"[{batch_idx}] Batch: {batch.batch_size}")
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value network which is updated in the inner loop
        with torch.no_grad():
            try:
                self.advantage_module(batch)
            except RuntimeError as ex:
                raise RuntimeError(f"{ex}\n{batch}") from ex
        loss = torch.tensor(0.0).to(self.device)
        n: int = self.frames_per_batch // self.sub_batch_size
        assert (
            n > 0
        ), f"frames_per_batch({self.frames_per_batch}) // sub_batch_size({self.sub_batch_size}) = {n} should be > {0}."
        for _ in range(n):
            subdata: TensorDict = self.replay_buffer.sample(self.sub_batch_size)
            loss_vals: TensorDict = self.loss_module(subdata.to(self.device))
            loss, losses = self.loss(loss_vals, loss, tag)
        self.log_dict(losses)
        self.log(f"loss/{tag}", loss, prog_bar=True)
        reward: Tensor = batch["next", "reward"]
        self.log(f"reward/{tag}", reward.mean().item(), prog_bar=True)
        step_count: Tensor = batch["step_count"]
        self.log(f"step_count/{tag}", step_count.max().item(), prog_bar=True)
        return loss

    def loss(
        self,
        loss_vals: TensorDict,
        loss: torch.Tensor = None,
        tag: str = "train",
    ) -> ty.Tuple[torch.Tensor, ty.Dict[str, torch.Tensor]]:
        """Updates the input loss and extracts losses from input `TensorDict` and collects them into a dict."""
        if loss is None:
            loss = torch.tensor(0.0).to(loss_vals.device)
        loss_dict: ty.Dict[str, torch.Tensor] = {}
        for key, value in loss_vals.items():
            if "loss_" in key:
                loss = loss + value
                logger.trace(f"key: {value}")
                loss_dict[f"{key}/{tag}"] = value
        assert isinstance(loss, torch.Tensor)
        return loss, loss_dict

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

    def transformed_env(self, base_env: EnvBase) -> EnvBase:
        """Setup transformed environment."""
        return base_env

    def _init_env(self, env: ty.Union[str, EnvBase], **kwargs: ty.Any) -> EnvBase:
        """Utility function to init an env.

        Args:
            env (ty.Union[str, EnvBase]): _description_

        Returns:
            EnvBase: _description_
        """
        if isinstance(env, str):
            env = GymEnv(
                env,
                device=kwargs.get("device", "cpu"),
                frame_skip=kwargs.get("frame_skip", 1),
            )
        return env
