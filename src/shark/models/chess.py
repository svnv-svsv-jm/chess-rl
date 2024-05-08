__all__ = ["BaseChess"]

from loguru import logger
import typing as ty

from torchrl.envs import EnvBase

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
        """
        Args:
            engine_executable (str, optional):
                Path to the chess engine executable. Defaults to `None`.

            n_mlp_layers (int, optional):
                Number of MLP layers. Defaults to `3`.

            num_mlp_cells (ty.Sequence[int] | int, optional):
                Number of MLP cells (MLP hidden dimension). Defaults to `256`.

            depth (int, optional):
                Depth for the CNN layers. Defaults to `3`.

            num_cells (ty.Sequence[int] | int, optional):
                Number of CNN cells (CNN hidden dimension). Defaults to `256`.

            kernel_sizes (ty.Sequence[int | ty.Sequence[int]] | int, optional):
                Kernel size for the CNN layers. Defaults to `3`.

            strides (ty.Sequence[int] | int, optional):
                Strides for the CNN layers. Defaults to `1`.

            paddings (ty.Sequence[int] | int, optional):
                Paddings for the CNN layers. Defaults to `0`.

            critic_action_hidden_dim (int, optional):
                Hidden dimension for the action critic. Defaults to `32`.

            env_kwargs (ty.Dict[str, ty.Any], optional):
                Key-word arguments for the chess environment. Defaults to `{}`.

            critic_type (str, optional):
                Type of critic.
                Possible choices include: `"observation"`, `"observation-action"`.
                Defaults to `"observation"`.
        """
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
