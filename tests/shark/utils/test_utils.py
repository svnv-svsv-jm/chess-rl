import pytest
from loguru import logger
import typing as ty
import sys

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger

from shark.utils import find_device, nb_init, get_logged_metrics_from_trainer


def test_utils_for_cov() -> None:
    """Coverage."""
    nb_init()
    device = find_device()
    logger.info(device)
    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=2,
        val_check_interval=1,
        log_every_n_steps=1,
        logger=CSVLogger(
            save_dir="pytest_artifacts",
            name="name",
        ),
    )
    try:
        get_logged_metrics_from_trainer(trainer, from_dict=True)
    except Exception as e:
        pass


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
