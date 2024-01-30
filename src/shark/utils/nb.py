# pylint: disable=unused-import
# pylint: disable=unused-argument
__all__ = ["nb_init"]

import os, sys
from loguru import logger
import warnings
import pyrootutils


def nb_init(logger_level: str = "INFO", add_std: bool = False) -> None:
    """Disables warnings and correctly finds the root directory of the project. Very handy when running notebooks.
    Args:
        logger_level (str): 'INFO'
            Logging level for loguru's logger.
    """
    warnings.filterwarnings("ignore")
    root = pyrootutils.setup_root(
        search_from=".",
        indicator=[".git", "pyproject.toml"],
        pythonpath=True,
        dotenv=True,
        cwd=True,
    )
    os.chdir(root)
    logger.remove()
    logger.add(sys.stderr, level=logger_level, format="{level} | {function} | {message}")
    logger.info(f"Set current dir to {os.path.basename(root)}")
    logger.info(f"You are using Python {sys.version}")
    logger.debug("You will see DEBUG messages.")
