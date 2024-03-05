from loguru import logger
import sys, os
import typing as ty

import pyrootutils
from omegaconf import DictConfig
import hydra
import lightning.pytorch as pl

from shark.utils import init_experiment, get_optimized_metric

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["pyproject.toml", "setup.py"],
    pythonpath=True,
    dotenv=True,
)
HERE = os.path.dirname(__file__)


@hydra.main(
    config_path=os.path.join(ROOT, HERE, "configs"),
    config_name="main",  # change using the flag `--config-name`
    version_base=None,
)
def main(cfg: DictConfig = None) -> float:
    """Train model. You can pass a different configuration from the command line as follows:
    >>> python main.py --config-name <name>
    """
    # config is relative to a module
    assert cfg is not None
    # Get model and trainer
    model, trainer = init_experiment(cfg)
    # Debug
    pl.Trainer(fast_dev_run=True).fit(model)
    # Train
    trainer.fit(model)
    # Get metric to optimizer
    metric_name: str = f"{cfg.optimize_metric}"
    metric = get_optimized_metric(trainer, optimize_metric=metric_name)
    # Return any number
    return metric


if __name__ == "__main__":
    """You can pass a different configuration from the command line as follows:
    >>> python main.py --config-name <name>
    """
    main()
