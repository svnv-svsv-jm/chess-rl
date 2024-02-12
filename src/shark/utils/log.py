__all__ = ["get_logged_metrics_from_trainer"]

import os
from loguru import logger
import lightning as L
import pandas as pd


def get_logged_metrics_from_trainer(trainer: L.Trainer, from_dict: bool = False) -> pd.DataFrame:
    """Gets logged metrics from trainer.

    Args:
        trainer (L.Trainer):
            `Trainer` object.
        from_dict (bool, optional):
            Whether to check directly for a `.csv` in the log directory or not.

    Returns:
        pd.DataFrame: Metrics.
    """
    if not from_dict:
        # Get metrics
        logs = trainer.logged_metrics
        logger.debug(logs)
        if isinstance(logs, dict):
            try:
                df = pd.DataFrame(logs)
            except ValueError as ex:
                logger.trace(ex)
                df = pd.DataFrame(logs, index=[0])
            return df
    # Look for CSV file: works only for CSVLogger
    log_dir = trainer.log_dir
    logger.debug(log_dir)
    assert isinstance(log_dir, str)
    filename = os.path.join(log_dir, "metrics.csv")
    df = pd.read_csv(filename)
    logger.debug(df.head())
    return df
