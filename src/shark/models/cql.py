__all__ = ["CQL", "CQLChess"]

import typing as ty
from loguru import logger

from ._base import BaseRL
from .chess import BaseChess


class CQL(BaseRL):
    """Base for CQL Model. See: https://pytorch.org/rl/tutorials/coding_ppo.html#training-loop"""

    def __init__(
        self,
        **kwargs: ty.Any,
    ) -> None:
        super().__init__(model="cql", **kwargs)  # pragma: no cover


class CQLChess(BaseChess):
    """Same but overrides the `transformed_env` method."""

    def __init__(self, **kwargs: ty.Any) -> None:
        super().__init__(model="cql", **kwargs)
