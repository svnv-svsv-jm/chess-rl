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

from shark.env import ChessEnv
from ._base import BaseRL


class BaseChess(BaseRL):
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
        env_kwargs: ty.Dict[str, ty.Any] = {},
        **kwargs: ty.Any,
    ) -> None:
        """Init."""
        self.engine_executable = engine_executable
        self.env_kwargs = env_kwargs.copy()
        base_env = ChessEnv(engine_executable, **self.env_kwargs)
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
            MLP(out_features=2 * out_features, **mlp_kwargs),
        )
        value_nn = torch.nn.Sequential(
            ConvNet(**cnn_kwargs),
            MLP(out_features=1, **mlp_kwargs),
        )
        super().__init__(
            actor_nn=actor_nn,
            value_nn=value_nn,
            **kwargs,
        )
        self.env_name = f"{ChessEnv.__name__}"

    def make_env(self) -> EnvBase:
        return ChessEnv(self.engine_executable, **self.env_kwargs)

    def transformed_env(self, base_env: EnvBase) -> EnvBase:
        """Setup transformed environment."""
        # return base_env
        env = TransformedEnv(
            base_env,
            transform=Compose(
                StepCounter(),
            ),
        )
        return env
