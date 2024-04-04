import pytest
from loguru import logger
import typing as ty
import sys, os

import torch
from tensordict import TensorDict
from torchrl.modules import MLP, QValueActor
from torchrl.data import OneHotDiscreteTensorSpec
from torchrl.objectives import DiscreteCQLLoss


def test_discretecql_website_example() -> None:
    """Test website's example."""
    n_obs, n_act = 4, 3
    value_net = MLP(in_features=n_obs, out_features=n_act)
    spec = OneHotDiscreteTensorSpec(n_act)
    actor = QValueActor(value_net, in_keys=["observation"], action_space=spec)
    loss_module = DiscreteCQLLoss(actor, action_space=spec)
    batch = [10]
    data = TensorDict(
        {
            "observation": torch.randn(*batch, n_obs),
            "action": spec.rand(batch),
            ("next", "observation"): torch.randn(*batch, n_obs),
            ("next", "done"): torch.zeros(*batch, 1, dtype=torch.bool),
            ("next", "terminated"): torch.zeros(*batch, 1, dtype=torch.bool),
            ("next", "reward"): torch.randn(*batch, 1),
        },
        batch,
    )
    loss = loss_module(data)
    logger.info(f"Loss: {loss}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s"])
