__all__ = ["CQLCritic"]

from loguru import logger
import torch
from torch import Tensor
from torchrl.modules import MLP, ConvNet


class CQLCritic(torch.nn.Module):
    """CQLCritic layer."""

    def __init__(
        self,
        action_hidden_dim: int = 32,
        mlp_kwargs: dict = {},
        cnn_kwargs: dict = {},
    ) -> None:
        """
        Args:
            mlp_kwargs (dict, optional): _description_. Defaults to {}.
            cnn_kwargs (dict, optional): _description_. Defaults to {}.
        """
        super().__init__()
        self.obs_net = torch.nn.Sequential(
            ConvNet(**cnn_kwargs),
            MLP(out_features=action_hidden_dim, **mlp_kwargs),
        )
        self.act_net = MLP(out_features=action_hidden_dim, **mlp_kwargs)
        self.mlp = MLP(out_features=1, **mlp_kwargs)

    def forward(self, observation: Tensor, action: Tensor) -> Tensor:
        """CQLCritic."""
        bs = action.size(0)
        logger.trace(f"action: {action.size()} - observation: {observation.size()}")
        h_obs: Tensor = self.obs_net(observation)
        h_act: Tensor = self.act_net(action)
        logger.trace(f"h_obs: {h_obs.size()} - h_act: {h_act.size()}")
        h: Tensor = torch.cat([h_obs, h_act], -1)
        h = self.mlp(h)
        return h
