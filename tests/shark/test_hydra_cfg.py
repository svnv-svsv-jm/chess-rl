# pylint: disable=no-member
import pytest
from loguru import logger
import sys, os
import typing as ty

from omegaconf import DictConfig
import hydra
from hydra import initialize_config_dir, compose
import pyrootutils
import lightning.pytorch as pl

from shark.models import PPOChess
from shark.utils.run import init_experiment, get_optimized_metric

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["pyproject.toml", "setup.py"],
    pythonpath=True,
    dotenv=True,
)
CONFIG_NAME = "test"


def test_yaml_config() -> None:
    """Tests we can create an experiment (model + trainer + callbacks etc.) from the Hydra config."""
    if __file__ not in sys.argv[0]:
        load_and_run_compose_api()
    else:
        load_and_run_hydra_main()


@hydra.main(
    version_base=None,
    config_path=os.path.join(ROOT, "configs"),
    config_name=CONFIG_NAME,
)
def load_and_run_hydra_main(cfg: DictConfig = None) -> ty.Optional[float]:
    """Hydra main call."""
    return _load_and_run(cfg)


def load_and_run_compose_api(cfg: DictConfig = None) -> ty.Optional[float]:
    """Hydra Compose API."""
    return _load_and_run(cfg)


def _load_and_run(cfg: DictConfig = None) -> ty.Optional[float]:
    """Load configuration and run a training."""
    if cfg is None:
        with initialize_config_dir(config_dir=os.path.join(ROOT, "configs")):
            # config is relative to a module
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    "hydra.sweeper.n_trials=1",
                    "trainer.max_steps=3",
                    "trainer.val_check_interval=1",
                    "trainer.log_every_n_steps=1",
                    "model.frames_per_batch=2",
                    "model.total_frames=10",
                ],
            )
    assert cfg is not None
    # Get model and trainer
    model, trainer = init_experiment(cfg, raise_error_on_hydra_shit=False)
    assert isinstance(model, PPOChess)
    assert isinstance(trainer, pl.Trainer)
    # Train
    trainer.fit(model)
    # Get metric to optimizer
    metric_name: str = f"{cfg.optimize_metric}"
    metric = get_optimized_metric(trainer, optimize_metric=metric_name)
    # Return any number
    return metric


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    pytest.main([__file__, "-x", "-s", "--pylint"])
