__all__ = ["init_experiment", "get_optimized_metric"]

from loguru import logger
import typing as ty
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import UnsupportedInterpolationType
import hydra
from hydra import initialize, initialize_config_dir, compose
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.loggers import Logger


_OUT_METRIC = ty.Union[torch.Tensor, ty.Dict[str, torch.Tensor]]


def init_experiment(cfg: DictConfig) -> ty.Tuple[pl.LightningModule, pl.Trainer]:
    """Initializes model and trainer.

    Args:
        cfg (DictConfig): Hydra config.

    Returns:
        pl.LightningModule: model.
        pl.Trainer: trainer.
    """
    # Model
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model, _convert_="all")
    # Loggers
    loggers: ty.Dict[str, Logger] = hydra.utils.instantiate(cfg.loggers, _convert_="all")
    logs = list(loggers.values())
    # Callbacks
    callbacks: ty.Dict[str, Callback] = hydra.utils.instantiate(cfg.callbacks, _convert_="all")
    cbs = list(callbacks.values())
    # Trainer
    # trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, _convert_="all")
    trainer: dict = OmegaConf.to_container(cfg.trainer)  # type: ignore
    trainer.pop("logger", None)
    trainer.pop("callbacks", None)
    trainer.pop("default_root_dir", None)
    try:
        default_root_dir = cfg.paths.output_dir
    except UnsupportedInterpolationType:
        default_root_dir = None
    pl_trainer = pl.Trainer(
        default_root_dir=default_root_dir,
        logger=logs if len(logs) > 0 else True,
        callbacks=cbs if len(cbs) > 0 else None,  # type: ignore
        **trainer,
    )
    return model, pl_trainer


def get_optimized_metric(trainer: pl.Trainer, optimize_metric: str) -> float:
    """Extracts metric to be optimized in Hydra's HPO from logged metrics in Trainer.
    Metric needs to be logged in the training process or this will fail.

    Args:
        trainer (pl.Trainer): Trainer.

        optimize_metric (str): Name of the metric to be optimized.

    Returns:
        float: Metric value, converted to float.
    """
    logged_metrics = trainer.logged_metrics
    metric: _OUT_METRIC = logged_metrics[optimize_metric]
    # Ok if numerical
    if isinstance(metric, (int, float)):
        return float(metric)
    # If Tensor, convert to float
    assert isinstance(
        metric, torch.Tensor
    ), f"This is currently only supported for Tensor metrics but got {metric}."
    output = float(metric.mean())
    return output
