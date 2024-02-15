# pylint: disable=no-member
import pytest
from loguru import logger
import sys, os
import typing as ty

from omegaconf import DictConfig
import hydra
from hydra import initialize, initialize_config_dir, compose
import pyrootutils
import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback

from shark.models import PPOChess

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["pyproject.toml", "setup.py"],
    pythonpath=True,
    dotenv=True,
)


def test_yaml_config() -> None:
    """Tests we can create an experiment (model + trainer + callbacks etc.) from the Hydra config."""
    if __file__ not in sys.argv[0]:
        pytest.skip("Skipping this when running all tests.")
    logger.info("Testing for config.yaml files...")
    with initialize_config_dir(config_dir=os.path.join(ROOT, "configs")):
        # config is relative to a module
        cfg = compose(config_name="main", overrides=["hydra.sweeper.n_trials=1"])
        assert cfg is not None
        # Model
        model = hydra.utils.instantiate(cfg.model, _convert_="all")
        assert isinstance(model, PPOChess)
        # Trainer
        trainer = hydra.utils.instantiate(cfg.trainer, _convert_="all")
        assert isinstance(trainer, pl.Trainer)
        # Callbacks
        cbs: ty.Dict[str, Callback] = hydra.utils.instantiate(cfg.callbacks, _convert_="all")
        logger.info(f"{type(cbs)}: {cbs}")
        for _, callback in cbs.items():
            assert isinstance(callback, Callback)
        # Return any number
        return 0


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    pytest.main([__file__, "-x", "-s", "--pylint"])
