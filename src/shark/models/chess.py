__all__ = ["BaseChess"]

from loguru import logger
import typing as ty

import torch
from torchrl.envs import (
    Compose,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs import EnvBase
from torchrl.modules import MLP, ConvNet

from shark.env import ChessEnv, make_chess_env
from ._base import BaseRL
from .utils import make_chess_actor_critic


class BaseChess(BaseRL):
    """Same but overrides the `transformed_env` method."""

    def __init__(
        self,
        engine_executable: str = None,
        n_mlp_layers: int = 3,
        num_mlp_cells: ty.Sequence[int] | int = 256,
        depth: int = 3,
        num_cells: ty.Sequence[int] | int = 256,
        kernel_sizes: ty.Sequence[int | ty.Sequence[int]] | int = 3,
        strides: ty.Sequence[int] | int = 1,
        paddings: ty.Sequence[int] | int = 0,
        critic_action_hidden_dim: int = 32,
        env_kwargs: ty.Dict[str, ty.Any] = {},
        critic_type: str = "observation",
        **kwargs: ty.Any,
    ) -> None:
        """Init."""
        assert isinstance(engine_executable, str)
        self.engine_executable = engine_executable
        self.env_kwargs = env_kwargs.copy()
        base_env = ChessEnv(engine_executable, **self.env_kwargs)
        actor_nn, value_nn = make_chess_actor_critic(
            base_env=base_env,
            num_cells=num_cells,
            n_mlp_layers=n_mlp_layers,
            num_mlp_cells=num_mlp_cells,
            kernel_sizes=kernel_sizes,
            depth=depth,
            paddings=paddings,
            strides=strides,
            critic_action_hidden_dim=critic_action_hidden_dim,
            critic_type=critic_type,
        )
        super().__init__(
            actor_nn=actor_nn,
            value_nn=value_nn,
            discrete=True,
            qvalue_actor=True,
            **kwargs,
        )
        self.env_name = f"{ChessEnv.__name__}"

    def make_env(self) -> EnvBase:
        return make_chess_env(self.engine_executable, **self.env_kwargs)

    # def transformed_env(self, base_env: EnvBase) -> EnvBase:
    #     """Setup transformed environment."""
    #     # return base_env
    #     env = TransformedEnv(
    #         base_env,
    #         transform=Compose(
    #             StepCounter(),
    #         ),
    #     )
    #     return env
