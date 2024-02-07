__all__ = ["MLP", "block"]

from loguru import logger
import typing as ty
import numpy as np
import torch


def block(
    out_features: int,
    normalize: bool = False,
    negative_slope: float = 0.2,
    batch_norm_eps: float = 0.8,
    leaky_relu: bool = False,
    tanh: bool = False,
    dropout: bool = False,
    p: float = 0.1,
) -> ty.List[torch.nn.Module]:
    """Creates a small neural block.
    Args:
        out_features (int):
            Output dimension.
        normalize (bool, optional):
            Whether to use Batch 1D normalization. Defaults to True.
        negative_slope (float, optional):
            Negative slope for ReLU layers. Defaults to 0.2.
        batch_norm_eps (float, optional):
            Epsilon for Batch 1D normalization. Defaults to 0.8.
        dropout (bool, optional):
            Whether to add a Dropout layer.
    """
    layers: ty.List[torch.nn.Module] = []
    layers.append(torch.nn.LazyLinear(out_features))
    # Normalization
    if normalize:
        layers.append(torch.nn.BatchNorm1d(out_features, batch_norm_eps))  # type: ignore
    # Non-linear function
    if leaky_relu:
        layers.append(torch.nn.LeakyReLU(negative_slope, inplace=True))  # type: ignore
    elif tanh:
        layers.append(torch.nn.Tanh())
    else:
        layers.append(torch.nn.ReLU())
    # Dropout
    if dropout:
        layers.append(torch.nn.Dropout(p))
    return list(layers)


class MLP(torch.nn.Module):
    """General MLP class."""

    def __init__(
        self,
        out_features: ty.Union[int, ty.Sequence[int]],
        hidden_dims: ty.Sequence[int] = None,
        last_activation: torch.nn.Module = None,
        flatten: bool = True,
        flatten_start_dim: int = 0,
        expected_input_size: torch.Size = None,
        **kwargs: ty.Any,
    ) -> None:
        """General MLP.
        Args:
            out_features (ty.Union[int, ty.Sequence[int]]):
                Output dimension or shape.
            hidden_dims (ty.Sequence[int], optional):
                Sequence of hidden dimensions. Defaults to [].
            last_activation (torch.nn.Module, optional):
                Last activation for the MLP. Defaults to None.
            **kwargs (optional):
                See function :func:`~brainiac_2.nn.block`
        """
        super().__init__()
        self.flatten = flatten
        self.flatten_start_dim = flatten_start_dim
        # Sanitize
        if hidden_dims is None:
            hidden_dims = []
        else:
            for i, h in enumerate(hidden_dims):
                hidden_dims[i] = int(h)  # type: ignore
        if isinstance(out_features, int):
            out_features = [out_features]
        else:
            for i, h in enumerate(out_features):
                out_features[i] = int(h)  # type: ignore
        # Set up
        self.out_features = out_features
        out_shape = [out_features] if isinstance(out_features, int) else out_features
        layers = []
        if len(hidden_dims) > 0:
            for _, h in enumerate(hidden_dims):
                layers += block(h, **kwargs)
            layers.append(torch.nn.LazyLinear(int(np.prod(out_shape))))
        else:
            layers.append(torch.nn.LazyLinear(int(np.prod(out_shape))))
        if last_activation is not None:
            layers.append(last_activation)
        self.model = torch.nn.Sequential(*layers)
        # Private
        self._size = expected_input_size

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Basic forward pass."""
        observation = observation.float().to(self.device)
        logger.trace(f"observation ({observation.device}|{self.device}): {observation.size()}")
        if self._size is None:
            self._size = observation.size()
        if self.flatten:
            start_dim = self.flatten_start_dim
            if observation.dim() > len(self._size):
                start_dim += 1
            observation = observation.flatten(start_dim)
            logger.trace(f"observation ({observation.device}|{self.device}): {observation.size()}")
        output_tensor: torch.Tensor = self.model(observation)
        return output_tensor

    @property
    def device(self) -> torch.device:
        """Device..."""
        return next(self.parameters()).device
