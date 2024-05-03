__all__ = ["ToDevice"]

from loguru import logger
import torch


class ToDevice(torch.nn.Module):
    """Fucking cast to device."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self._cast_device = device
        self.__has_warned = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Casts to device."""
        if x.device != self._cast_device and not self.__has_warned:
            logger.warning(
                f"Casting from {x.device} to {self._cast_device} was necessary. Please check again your configuration."
            )
            self.__has_warned = True
        return x.to(self._cast_device)
