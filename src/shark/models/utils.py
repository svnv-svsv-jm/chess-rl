__all__ = ["make_chess_actor_critic", "initialize", "initialize_actor"]

import typing as ty
from loguru import logger
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.envs import EnvBase
from torchrl.modules import ProbabilisticActor, TanhNormal, QValueActor, ValueOperator, MLP, ConvNet
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss, CQLLoss, DiscreteCQLLoss, SoftUpdate

from shark.env import ChessEnv
from shark.nn import CQLCritic, ToDevice


def make_chess_actor_critic(
    base_env: ChessEnv,
    n_mlp_layers: int = 3,
    num_mlp_cells: ty.Sequence[int] | int = 256,
    depth: int = 3,
    num_cells: ty.Sequence[int] | int = 256,
    kernel_sizes: ty.Sequence[int | ty.Sequence[int]] | int = 3,
    strides: ty.Sequence[int] | int = 1,
    paddings: ty.Sequence[int] | int = 0,
    critic_action_hidden_dim: int = 32,
    critic_type: str = "ppo",
    out_features_multiplier: int = 2,
) -> ty.Tuple[torch.nn.Module, torch.nn.Module]:
    """Create an actor and a critic for the `ChessEnv`.

    Args:
        base_env (ChessEnv):
            Chess environment.

        num_cells (int): _description_

        num_mlp_cells (int): _description_

        n_mlp_layers (int): _description_

        depth (int): _description_

        kernel_sizes (ty.Sequence[int  |  ty.Sequence[int]] | int, optional): _description_. Defaults to 3.

        strides (ty.Sequence | int, optional): _description_. Defaults to 1.

        paddings (ty.Sequence | int, optional): _description_. Defaults to 0.

        critic_action_hidden_dim (int, optional): _description_. Defaults to 32.

        out_features_multiplier (int, optional): _description_. Defaults to 2.

    Raises:
        ValueError: If `model` is not recognized.

    Returns:
        ty.Tuple[torch.nn.Module, torch.nn.Module]: _description_
    """
    # Set up
    out_features = base_env.action_spec.shape[-1]
    if isinstance(num_cells, (float, int)):
        num_cells = int(num_cells)
    if isinstance(num_mlp_cells, (float, int)):
        num_mlp_cells = int(num_mlp_cells)
    mlp_kwargs = dict(
        depth=int(n_mlp_layers),
        num_cells=num_mlp_cells,
        dropout=True,
    )
    cnn_kwargs = dict(
        depth=int(depth),
        num_cells=num_cells,
        kernel_sizes=kernel_sizes,
        strides=strides,
        paddings=paddings,
    )
    # Actor
    actor_nn_ = torch.nn.Sequential(
        ConvNet(**cnn_kwargs),
        MLP(out_features=out_features_multiplier * out_features, **mlp_kwargs),
    ).to(base_env.device)
    actor_nn = torch.nn.Sequential(
        ToDevice(next(actor_nn_.parameters()).device),
        actor_nn_,
    )
    # Critic
    value_nn: torch.nn.Module
    critic_type = critic_type.lower()
    if critic_type in [
        "ppo",
        "state",
        "observation",
    ]:
        value_nn = torch.nn.Sequential(
            ConvNet(**cnn_kwargs),
            MLP(out_features=1, **mlp_kwargs),
        ).to(base_env.device)
    elif critic_type in [
        "state-action",
        "state_action",
        "observation-action",
        "observation_action",
    ]:
        value_nn = CQLCritic(
            action_hidden_dim=critic_action_hidden_dim,
            mlp_kwargs=mlp_kwargs,
            cnn_kwargs=cnn_kwargs,
        ).to(base_env.device)
    else:
        raise ValueError(f"Unrecognized critic type {critic_type}")
    # Return
    return actor_nn, value_nn


def initialize(
    policy_module: TensorDictModule,
    env: EnvBase,
    model: str,
    discrete: bool = True,
    value_nn: torch.nn.Module = None,
    flatten_state: bool = False,
    loss_function: str = "smooth_l1",
    alpha_init: float = 1,
    tau: float = None,
    gamma: float = None,
    lmbda: float = None,
    clip_epsilon: float = 0.2,
    entropy_bonus: bool = True,
    samples_mc_entropy: int = 1,
    entropy_coef: float = 0.01,
) -> ty.Dict[str, ty.Optional[TensorDictModule]]:
    """_summary_

    Args:
        policy_module (TensorDictModule): _description_
        env (EnvBase): _description_
        model (str): _description_
        discrete (bool, optional): _description_. Defaults to True.
        value_nn (torch.nn.Module, optional): _description_. Defaults to None.
        flatten_state (bool, optional): _description_. Defaults to False.
        loss_function (str, optional): _description_. Defaults to "smooth_l1".
        alpha_init (float, optional): _description_. Defaults to 1.
        tau (float, optional): _description_. Defaults to None.
        gamma (float, optional): _description_. Defaults to None.
        lmbda (float, optional): _description_. Defaults to None.
        clip_epsilon (float, optional): _description_. Defaults to 0.2.
        entropy_bonus (bool, optional): _description_. Defaults to True.
        samples_mc_entropy (int, optional): _description_. Defaults to 1.
        entropy_coef (float, optional): _description_. Defaults to 0.01.

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        ty.Dict[str, ty.Optional[TensorDictModule]]: _description_
    """
    policy_module = policy_module.to(env.device)
    target_net_updater = None
    value_module = None
    if model in ["cql"]:
        advantage_module = None
        # Loss CQL
        if discrete:
            loss_module = DiscreteCQLLoss(policy_module, action_space=env.action_spec)
        else:
            if value_nn is None:
                raise ValueError(f"`value_nn` must be {torch.nn.Module} when continuous CQL.")
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
        if value_nn is None:
            raise ValueError(f"`value_nn` must be {torch.nn.Module} when PPO.")
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
            entropy_bonus=entropy_bonus,
            entropy_coef=entropy_coef,
            samples_mc_entropy=samples_mc_entropy,
            # these keys match by default but we set this for completeness
            critic_coef=1.0,
            # gamma=0.99,
            loss_critic_type=loss_function,
        )
        loss_module.set_keys(value_target=advantage_module.value_target_key)
    else:
        raise ValueError(f"Unrecognized model {model}")
    return dict(
        loss_module=loss_module,
        advantage_module=advantage_module,
        value_module=value_module,
        target_net_updater=target_net_updater,
    )


def initialize_actor(
    actor_nn: torch.nn.Module,
    env: EnvBase,
    flatten_state: bool = False,
    qvalue: bool = False,
) -> TensorDictModule:
    # Cast to env device
    actor_nn = actor_nn.to(env.device)

    # Q-Value actor
    if qvalue:
        actor_net = torch.nn.Sequential(
            ToDevice(next(actor_nn.parameters()).device),
            actor_nn,
        )
        policy_module = QValueActor(
            actor_net,
            in_keys=["observation"],
            action_space=env.action_spec,
        )

    # No Q-Value
    else:
        action_space = env.action_spec
        out_features = action_space.shape[-1]
        logger.debug(f"MLP out_shape: {out_features}")
        actor_net = torch.nn.Sequential(
            ToDevice(next(actor_nn.parameters()).device),
            torch.nn.Flatten(0) if flatten_state else torch.nn.Identity(),
            actor_nn,
            NormalParamExtractor(),
        )
        logger.debug(f"Initialized actor: {actor_net}")
        tdm = TensorDictModule(
            actor_net,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        )
        td = env.reset()
        tdm = tdm.to(td.device)  # Cast to device
        tdm(td)  # pylint: disable=not-callable
        policy_module = ProbabilisticActor(
            module=tdm,
            spec=env.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "min": 0,  # env.action_spec.space.minimum,
                "max": 1,  # env.action_spec.space.maximum,
            },
            return_log_prob=True,  # we'll need the log-prob for the numerator of the importance weights
        )

    # Initialize and return
    td = env.reset()
    policy_module = policy_module.to(td.device)  # Cast to device
    policy_module(td)  # pylint: disable=not-callable
    return policy_module
