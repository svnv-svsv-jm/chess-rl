__all__ = ["make_chess_actor_critic", "initialize"]

import typing as ty
from loguru import logger
import torch
from tensordict.nn import TensorDictModule
from torchrl.envs import EnvBase
from torchrl.modules import ValueOperator, MLP, ConvNet
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss, CQLLoss, DiscreteCQLLoss, SoftUpdate

from shark.env import ChessEnv
from shark.nn import CQLCritic


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
    actor_nn = torch.nn.Sequential(
        ConvNet(**cnn_kwargs),
        MLP(out_features=out_features_multiplier * out_features, **mlp_kwargs),
    )
    value_nn: torch.nn.Module
    if critic_type in ["ppo", "state", "observation"]:
        value_nn = torch.nn.Sequential(
            ConvNet(**cnn_kwargs),
            MLP(out_features=1, **mlp_kwargs),
        )
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
        )
    else:
        raise ValueError(f"Unrecognized critic type {critic_type}")
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
