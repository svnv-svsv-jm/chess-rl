__all__ = ["PPO", "PPOPendulum", "PPOChess"]

from loguru import logger
import typing as ty

import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
    FlattenObservation,
)
from torchrl.envs import EnvBase
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, MLP, ConvNet
from torchrl.objectives import ClipPPOLoss as BuggedClipPPOLoss
from torchrl.objectives.value import GAE

from shark.env import ChessEnv
from shark.utils.patch import _cache_values
from ._base import BaseRL


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


class PPO(BaseRL):
    """Base for PPO Model. See: https://pytorch.org/rl/tutorials/coding_ppo.html#training-loop"""

    def __init__(
        self,
        actor_nn: torch.nn.Module,
        value_nn: torch.nn.Module,
        env: ty.Union[str, EnvBase] = "InvertedDoublePendulum-v4",
        gamma: float = 0.99,
        lmbda: float = 0.95,
        entropy_eps: float = 1e-4,
        clip_epsilon: float = 0.2,
        in_keys: ty.List[str] = ["observation"],
        flatten_state: bool = False,
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
        self.save_hyperparameters(
            ignore=[
                "base_env",
                "env",
                "loss_module",
                "policy_module",
                "value_module",
            ]
        )
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_eps = entropy_eps
        self.in_keys = in_keys
        # Environment
        base_env = self._init_env(env)
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
            in_keys=self.in_keys,
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
        # Critic
        value_net = torch.nn.Sequential(
            torch.nn.Flatten(1) if flatten_state else torch.nn.Identity(),
            value_nn,
        )
        logger.debug(f"Initialized critic: {value_net}")
        value_module = ValueOperator(
            module=value_net,
            in_keys=self.in_keys,
        )
        td = env.reset()
        value_module(td)
        # Loss
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
            loss_critic_type="smooth_l1",
        )
        loss_module.set_keys(value_target=advantage_module.value_target_key)
        # Call superclass
        super().__init__(
            env=env,
            loss_module=loss_module,
            policy_module=policy_module,
            value_module=value_module,
            advantage_module=advantage_module,
            in_keys=self.in_keys,
            **kwargs,
        )


class PPOPendulum(PPO):
    """Basic PPO Model. See: https://pytorch.org/rl/tutorials/coding_ppo.html#training-loop"""

    def __init__(
        self,
        env: ty.Union[str, EnvBase] = "InvertedDoublePendulum-v4",
        num_cells: int = 256,
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
        base_env = self._init_env(env, **kwargs)
        out_features = base_env.action_spec.shape[-1]
        actor_nn = MLP(
            out_features=2 * out_features,
            depth=n_mlp_layers,
            num_cells=num_cells,
            dropout=True,
        )
        value_nn = MLP(
            out_features=1,
            depth=n_mlp_layers,
            num_cells=num_cells,
            dropout=True,
        )
        # Call superclass
        super().__init__(
            env=env,
            actor_nn=actor_nn,
            value_nn=value_nn,
            **kwargs,
        )

    def transformed_env(self, base_env: EnvBase) -> EnvBase:
        """Setup transformed environment."""
        obs_norm = ObservationNorm(in_keys=self.in_keys)
        double2float = DoubleToFloat()
        # setattr(double2float, "transform_observation_spec", transform_observation_spec)
        env = TransformedEnv(
            base_env,
            transform=Compose(
                obs_norm,
                double2float,
                StepCounter(),
            ),
        )
        obs_norm.init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
        logger.debug(f"normalization constant shape: {obs_norm.loc.shape}")
        return env


class PPOChess(PPO):
    """Same but overrides the `transformed_env` method."""

    def __init__(
        self,
        engine_executable: str = None,
        n_mlp_layers: int = 3,
        num_mlp_cells: ty.Sequence | int = 256,
        depth: int = 3,
        num_cells: ty.Sequence | int = 256,
        kernel_sizes: ty.Sequence[int | ty.Sequence[int]] | int = 3,
        strides: ty.Sequence | int = 1,
        paddings: ty.Sequence | int = 0,
        use_one_hot: bool = True,
        **kwargs: ty.Any,
    ) -> None:
        """Init."""
        self.use_one_hot = use_one_hot
        base_env = ChessEnv(engine_executable)
        out_features = base_env.action_spec.shape[-1]
        mlp_kwargs = dict(
            depth=n_mlp_layers,
            num_cells=num_mlp_cells,
            dropout=True,
        )
        cnn_kwargs = dict(
            depth=depth,
            num_cells=num_cells,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
        )
        actor_nn = torch.nn.Sequential(
            ConvNet(**cnn_kwargs),
            MLP(out_features=2 * out_features, **mlp_kwargs),
        )
        value_nn = torch.nn.Sequential(
            ConvNet(**cnn_kwargs),
            MLP(out_features=1, **mlp_kwargs),
        )
        super().__init__(
            env=ChessEnv(engine_executable),
            actor_nn=actor_nn,
            value_nn=value_nn,
            **kwargs,
        )

    def transformed_env(self, base_env: EnvBase) -> EnvBase:
        """Setup transformed environment."""
        # return base_env
        env = TransformedEnv(
            base_env,
            transform=Compose(
                # FlattenObservation(
                #     first_dim=0,
                #     last_dim=-2 if self.use_one_hot else -1,
                #     in_keys=self.in_keys,
                #     allow_positive_dim=True,
                # ),
                StepCounter(),
            ),
        )
        return env
