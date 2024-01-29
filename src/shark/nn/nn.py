__all__ = ["MLP", "block"]

from loguru import logger
import typing as ty
import numpy as np
import torch


def block(
    in_features: int,
    out_features: int,
    normalize: bool = True,
    negative_slope: float = 0.2,
    batch_norm_eps: float = 0.8,
    leaky_relu: bool = True,
    dropout: bool = False,
) -> ty.List[torch.nn.Module]:
    """Creates a small neural block.
    Args:
        in_features (int):
            Input dimension.
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
    layers.append(torch.nn.Linear(in_features, out_features))
    if normalize:
        layers.append(torch.nn.BatchNorm1d(out_features, batch_norm_eps))  # type: ignore
    if leaky_relu:
        layers.append(torch.nn.LeakyReLU(negative_slope, inplace=True))  # type: ignore
    else:
        layers.append(torch.nn.ReLU())
    if dropout:
        layers.append(torch.nn.Dropout())
    return list(layers)


class MLP(torch.nn.Module):
    """General MLP class."""

    def __init__(
        self,
        in_features: int,
        out_features: ty.Union[int, ty.Sequence[int]],
        hidden_dims: ty.Sequence[int] = None,
        last_activation: torch.nn.Module = None,
        **kwargs: ty.Any,
    ) -> None:
        """General MLP.
        Args:
            in_features (int):
                Input dimension or shape.
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
        # Sanitize
        in_features = int(in_features)
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
        layers_dims = [in_features, *hidden_dims]
        if len(hidden_dims) > 0:
            for i in range(0, len(layers_dims) - 1):
                layers += block(layers_dims[i], layers_dims[i + 1], **kwargs)
            layers.append(torch.nn.Linear(layers_dims[-1], int(np.prod(out_shape))))
        else:
            layers.append(torch.nn.Linear(in_features, int(np.prod(out_shape))))
        if last_activation is not None:
            layers.append(last_activation)
        self.model = torch.nn.Sequential(*layers)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Basic forward pass."""
        logger.trace(f"input_tensor: {input_tensor.size()}")
        output_tensor: torch.Tensor = self.model(input_tensor)
        return output_tensor
