__all__ = ["ConcatLayer"]

import torch
from torch import Tensor


class ConcatLayer(torch.nn.Module):
    """Concat layer."""

    def forward(self, observation: Tensor, action: Tensor) -> Tensor:
        """Concatenate input tensors."""
        return torch.cat([observation, action], -1)
