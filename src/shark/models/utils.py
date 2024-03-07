__all__ = ["make_chess_actor_critic"]

import typing as ty
import torch
from torchrl.modules import MLP, ConvNet

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
