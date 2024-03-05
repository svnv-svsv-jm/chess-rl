__all__ = ["BaseRL"]

from loguru import logger
import typing as ty

import lightning.pytorch as pl
import lightning.pytorch.callbacks as cb
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig, LRSchedulerConfigType
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.fabric.utilities.types import LRScheduler
import torch
from torch import Tensor
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.envs import ParallelEnv, EnvBase, EnvCreator, GymEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives.value import GAE
from torchrl.objectives import (
    ClipPPOLoss as BuggedClipPPOLoss,
    CQLLoss as BuggedCQLLoss,
    SoftUpdate,
)

from shark.datasets import CollectorDataset
from shark.utils.patch import step_and_maybe_reset, _cache_values


class RLTrainingLoop(pl.LightningModule):
    """RL training loop. See: https://pytorch.org/rl/tutorials/coding_ppo.html#training-loop"""

    def __init__(
        self,
        loss_module: TensorDictModule,
        policy_module: TensorDictModule,
        value_module: TensorDictModule,
        target_net_updater: SoftUpdate = None,
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
        use_checkpoint_callback: bool = False,
        save_every_n_train_steps: int = 100,
        raise_error_on_nan: bool = False,
        num_envs: int = 1,
        env_kwargs: ty.Dict[str, ty.Any] = {},
    ) -> None:
        """
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
            num_envs (int, optional): _description_. Defaults to 1.
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "base_env",
                "env",
                "loss_module",
                "policy_module",
                "value_module",
                "advantage_module",
                "target_net_updater",
            ]
        )
        if not hasattr(self, "env_kwargs"):
            self.env_kwargs = env_kwargs
        self.raise_error_on_nan = raise_error_on_nan
        self.use_checkpoint_callback = use_checkpoint_callback
        self.save_every_n_train_steps = save_every_n_train_steps
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.frame_skip = frame_skip
        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames
        self.sub_batch_size = sub_batch_size
        self.rollout_max_steps = rollout_max_steps
        self.num_envs = num_envs
        # Environment
        self.env = ParallelEnv(
            num_workers=num_envs,
            create_env_fn=EnvCreator(self._make_env),
            serial_for_single=True,
        )
        # Patch this method with your function
        self.env.step_and_maybe_reset = lambda arg: step_and_maybe_reset(self.env, arg)
        # Sanity check
        logger.debug(f"observation_spec: {self.env.observation_spec}")
        logger.debug(f"reward_spec: {self.env.reward_spec}")
        logger.debug(f"done_spec: {self.env.done_spec}")
        logger.debug(f"action_spec: {self.env.action_spec}")
        logger.debug(f"state_spec: {self.env.state_spec}")
        # Modules
        self.loss_module = loss_module
        self.policy_module = policy_module
        self.value_module = value_module
        self.target_net_updater = target_net_updater
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
        _dataset = getattr(self, "_dataset", None)
        if not isinstance(_dataset, CollectorDataset):
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
        self._dataset = None  # type: ignore
        return self.dataset

    def configure_callbacks(self) -> ty.Sequence[pl.Callback]:
        """Configure checkpoint."""
        callbacks = []
        if self.use_checkpoint_callback:
            ckpt_cb = cb.ModelCheckpoint(
                monitor="loss/train",
                mode="min",
                save_top_k=3,
                save_last=True,
                save_on_train_epoch_end=True,
                every_n_train_steps=self.save_every_n_train_steps,
            )
            callbacks.append(ckpt_cb)
        return callbacks

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

    def on_validation_epoch_start(self) -> None:
        """Validation step."""
        self.rollout()

    def on_test_epoch_start(self) -> None:
        """Test step."""
        self.rollout()

    def training_step(self, batch: TensorDict, batch_idx: int) -> Tensor:
        """Implementation follows the PyTorch tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html"""
        # Run optimization step
        loss = self.step(batch, batch_idx=batch_idx, tag="train")
        # This will run only if manual optimization
        self.manual_optimization_step(loss)
        # Update target network
        if isinstance(self.target_net_updater, SoftUpdate):
            self.target_net_updater.step()
        # We evaluate the policy once every `sefl.trainer.val_check_interval` batches of data
        n = self.trainer.val_check_interval
        if n is None:
            n = 10  # pragma: no cover
        n = int(n)
        if batch_idx % n == 0:
            self.rollout()
        # Return loss
        return loss

    def on_train_batch_end(
        self,
        outputs: Tensor | ty.Mapping[str, ty.Any] | None,
        batch: ty.Any,
        batch_idx: int,
    ) -> None:
        """Check if we have to stop. For some reason, Lightning can't understand this. Probably because we are using an `IterableDataset`."""
        # Stop on max steps
        global_step = self.trainer.global_step
        max_steps = self.trainer.max_steps
        if global_step >= max_steps:
            self.stop(f"global_step={global_step} > max_steps={max_steps}")
            return
        # Stop on max epochs
        current_epoch = self.trainer.current_epoch
        max_epochs = self.trainer.max_epochs
        if isinstance(max_epochs, int) and max_epochs > 0 and current_epoch >= max_epochs:
            self.stop(f"current_epoch={current_epoch} > max_epochs={max_epochs}")
            return
        # Stop on total frames
        if global_step >= self.total_frames:
            self.stop(f"global_step={global_step} > total_frames={self.total_frames}")
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

    def advantage(self, batch: ty.Any) -> None:
        """Advantage step."""

    def step(
        self,
        batch: TensorDict,
        batch_idx: int = None,
        tag: str = "train",
    ) -> Tensor:
        """Common step."""
        logger.trace(f"[{batch_idx}] Batch: {batch.batch_size}")
        self.advantage(batch)
        loss = torch.tensor(0.0).to(self.device)
        n: int = self.frames_per_batch // self.sub_batch_size
        assert (
            n > 0
        ), f"frames_per_batch({self.frames_per_batch}) // sub_batch_size({self.sub_batch_size}) = {n} should be > {0}."
        for _ in range(n):
            subdata: TensorDict = self.replay_buffer.sample(self.sub_batch_size)
            loss_vals: TensorDict = self.loss(subdata.to(self.device))
            loss, losses = self.collect_loss(loss_vals, loss, tag)
        self.log_dict(losses)
        self.log(f"loss/{tag}", loss, prog_bar=True)
        reward: Tensor = batch["next", "reward"]
        self.log(f"reward/{tag}", reward.mean().item(), prog_bar=True)
        step_count: Tensor = batch["step_count"]
        self.log(f"step_count/{tag}", step_count.max().item(), prog_bar=True)
        return loss

    def loss(self, data: TensorDict) -> TensorDict:
        """Evaluates the loss over input data."""
        loss_vals: TensorDict = self.loss_module(data.to(self.device))
        return loss_vals

    def collect_loss(
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
                if self.raise_error_on_nan:
                    assert not value.isnan().any(), f"Invalid loss value for {key}: {value}."
                loss = loss + value
                logger.trace(f"{key}: {value}")
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

    def make_env(self) -> EnvBase:
        """You have to implement this method, which has to take no inputs and return your environment."""
        raise NotImplementedError("You must implement this method.")

    def _make_env(self) -> EnvBase:
        """Lambda function."""
        env = self.make_env()
        return self.transformed_env(env)

    def state_dict(  # type: ignore
        self,
        *args: ty.Any,
        destination: ty.Dict[str, ty.Any] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> ty.Dict[str, ty.Any]:
        """State dict."""
        logger.trace(
            f"Calling with {args}; destination={destination}; prefix={prefix}; keep_vars={keep_vars}"
        )
        # Remove env (especially if Serial or Parallel and not plain BaseEnv)
        # Torch is unable to pickle it
        env = self.env
        self.env = None
        # Now return whatever Torch wanted us to return
        try:
            if destination is not None:
                return super().state_dict(
                    *args,
                    destination=destination,
                    prefix=prefix,
                    keep_vars=keep_vars,
                )
            return super().state_dict(
                *args,
                prefix=prefix,
                keep_vars=keep_vars,
            )
        # Bring `env` back
        finally:
            self.env = env


class ClipPPOLoss(BuggedClipPPOLoss):
    """Let's patch this."""

    def __init__(self, *args: ty.Any, **kwargs: ty.Any) -> None:
        super().__init__(*args, **kwargs)
        # self.__custom_dict__: ty.Dict[str, ty.Any] = {}
        # self.__dict__["_cache"] = {}

    @property
    @_cache_values
    def _cached_critic_network_params_detached(self) -> ty.Any:
        if not self.functional:
            return None
        return self.critic_network_params.detach()

    # @property
    # def __dict__(self) -> ty.Dict[str, ty.Any]:
    #     if "_cache" not in self.__custom_dict__:
    #         self.__custom_dict__["_cache"] = {}
    #     return self.__custom_dict__

    # @__dict__.setter
    # def __dict__(self, value: ty.Dict[str, ty.Any]) -> None:
    #     assert isinstance(value, dict)
    #     self.__custom_dict__ = value


class CQLLoss(BuggedCQLLoss):
    """Let's patch this."""

    def __init__(self, *args: ty.Any, **kwargs: ty.Any) -> None:
        super().__init__(*args, **kwargs)
        # self.__custom_dict__: ty.Dict[str, ty.Any] = {}
        # self.__dict__["_cache"] = {}

    @property
    @_cache_values
    def _cached_detach_qvalue_params(self) -> ty.Any:
        return self.qvalue_network_params.detach()


class BaseRL(RLTrainingLoop):
    """Base for RL Model. See: https://pytorch.org/rl/tutorials/coding_ppo.html#training-loop"""

    def __init__(
        self,
        actor_nn: torch.nn.Module,
        value_nn: torch.nn.Module,
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
        # Sanity check
        logger.debug(f"observation_spec: {base_env.observation_spec}")
        logger.debug(f"reward_spec: {base_env.reward_spec}")
        logger.debug(f"done_spec: {base_env.done_spec}")
        logger.debug(f"action_spec: {base_env.action_spec}")
        logger.debug(f"state_spec: {base_env.state_spec}")
        # Actor
        shape = base_env.observation_spec["observation"].shape
        assert isinstance(shape, torch.Size)
        out_features = base_env.action_spec.shape[-1]
        logger.debug(f"MLP out_shape: {out_features}")
        actor_net = torch.nn.Sequential(
            torch.nn.Flatten(0) if flatten_state else torch.nn.Identity(),
            actor_nn,
            NormalParamExtractor(),
        )
        logger.debug(f"Initialized actor: {actor_net}")
        policy_module = TensorDictModule(
            actor_net,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        )
        td = env.reset()
        policy_module(td)
        policy_module = ProbabilisticActor(
            module=policy_module,
            spec=env.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "min": 0,  # env.action_spec.space.minimum,
                "max": 1,  # env.action_spec.space.maximum,
            },
            return_log_prob=True,  # we'll need the log-prob for the numerator of the importance weights
        )
        logger.debug(f"Initialized policy: {policy_module}")
        # Critic and loss depend on the model
        target_net_updater = None
        if model in ["cql"]:
            advantage_module = None
            # Q-Value
            value_module = ValueOperator(
                module=value_nn,
                in_keys=["observation", "action"],
                out_keys=["state_action_value"],
            )
            td = env.reset()
            td = env.rand_action(td)
            td = env.step(td)
            td = value_module(td)
            logger.debug(f"Initialized value_module: {td}")
            # Loss CQL
            loss_module = CQLLoss(
                actor_network=policy_module,
                qvalue_network=value_module,
                action_spec=env.action_spec,
                alpha_init=alpha_init,
                loss_function=loss_function,
            )
            loss_module.make_value_estimator(gamma=gamma)
            target_net_updater = SoftUpdate(loss_module, tau=tau)
        elif model in ["ppo"]:
            # Value
            value_net = torch.nn.Sequential(
                torch.nn.Flatten(1) if flatten_state else torch.nn.Identity(),
                value_nn,
            )
            value_module = ValueOperator(
                module=value_net,
                in_keys=["observation"],
            )
            td = env.reset()
            value_module(td)
            # Loss PPO
            advantage_module = GAE(
                gamma=gamma,
                lmbda=lmbda,
                value_network=value_module,
                average_gae=True,
            )
            loss_module = ClipPPOLoss(
                actor=policy_module,
                critic=value_module,
                clip_epsilon=clip_epsilon,
                entropy_bonus=bool(entropy_eps),
                entropy_coef=entropy_eps,
                # these keys match by default but we set this for completeness
                critic_coef=1.0,
                # gamma=0.99,
                loss_critic_type=loss_function,
            )
            loss_module.set_keys(value_target=advantage_module.value_target_key)
        else:
            raise ValueError(f"Unrecognized model {model}")
        # Call superclass
        super().__init__(
            loss_module=loss_module,
            policy_module=policy_module,
            value_module=value_module,
            target_net_updater=target_net_updater,
            **kwargs,
        )
        self.advantage_module = advantage_module

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
