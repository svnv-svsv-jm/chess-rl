# pylint: disable=unused-argument
__all__ = ["Logger"]

import typing as ty
from loguru import logger
import sys, os
from datetime import datetime


class Logger:
    """This class allows you to print anything to both the terminal and a file."""

    def __init__(
        self,
        filename: str,
        mode: str = "a+",
        time_info: bool = True,
    ):
        self.terminal = sys.stdout
        self.filename = filename
        self.mode = mode
        self.time_info = time_info

    @logger.catch(level="TRACE")
    def write(self, text: str) -> None:
        """Write to sinks."""
        # datetime object containing current date and time
        if self.time_info:
            now = datetime.now()
            now_str = now.strftime("%d/%m/%Y %H:%M:%S")
            text = f"{now_str}  {text} "
        # write to outputs
        self.terminal.write(text)
        with open(self.filename, self.mode) as log:
            log.write(text)
            log.flush()  # If you want the output to be visible immediately

    @logger.catch(level="TRACE")
    def flush(
        self,
        *args: ty.Any,
        **kwargs: ty.Any,
    ) -> None:
        """This flush method is needed for Python3 compatibility. This handles the flush command by doing nothing. You might want to specify some extra behavior here."""
        with open(self.filename, self.mode) as log:
            log.flush()  # If you want the output to be visible immediately

    def isatty(self) -> bool:
        """This needs to exist."""
        return False
