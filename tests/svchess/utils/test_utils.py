import pytest
from loguru import logger
import typing as ty
import sys

import torch

from svchess.utils import find_device, nb_init


def test_nb() -> None:
    """Test."""
    nb_init()


def test_find_device() -> None:
    """Test."""
    device = find_device()
    logger.info(device)
    device = find_device("auto")
    logger.info(device)
    device = find_device(torch.device("cpu"))
    logger.info(device)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
