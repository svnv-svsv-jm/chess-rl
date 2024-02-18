__all__ = ["DebugCallback"]

from loguru import logger
import typing as ty

import torch
import lightning.pytorch as pl


class DebugCallback(pl.Callback):
    """Prints stuff at different moments during execution. Useful when debugging or during testing."""

    def __init__(self, level: str = "DEBUG") -> None:
        super().__init__()
        self.level = level.upper()

    def _log(self, msg: str) -> None:  # pragma: no cover
        """Logs to correct level."""
        if self.level == "INFO":
            logger.info(msg)
        elif self.level == "DEBUG":
            logger.debug(msg)
        elif self.level == "TRACE":
            logger.trace(msg)
        else:
            raise ValueError(f"Unknown level {self.level}")

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str = None,
    ) -> None:
        """Called when fit, validate, test, predict, or tune begins."""
        self._log(f"\n{pl_module}")

    def on_before_zero_grad(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
        *args: ty.Any,
    ) -> None:
        """Called before ``optimizer.zero_grad()``."""
        msg = self._get_optim_info(optimizer)
        self._log(f"Before zero grad. {msg}")

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
        **kwargs: ty.Any,
    ) -> None:
        """Called before ``optimizer.step()``."""
        msg = self._get_optim_info(optimizer)
        self._log(f"Before optimizer step. {msg}")

    def on_before_backward(
        self,
        *args: ty.Any,
    ) -> None:
        self._log("Before backward.")

    def on_after_backward(
        self,
        *args: ty.Any,
    ) -> None:
        self._log("After backward.")

    def on_train_batch_start(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: ty.Any,
        batch_idx: int,
        *args: ty.Any,
    ) -> None:
        self._batch_type_info(batch, batch_idx, "Training")

    def on_train_batch_end(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: ty.Any,
        batch: ty.Any,
        batch_idx: int,
        *args: ty.Any,
    ) -> None:
        self._log(f"On training batch end. Got the following outputs: {outputs}")
        if isinstance(outputs, dict) and "loss" in outputs.keys():
            loss = outputs["loss"]
            if isinstance(loss, torch.Tensor):
                self._log(f"Loss with value {loss} on device {loss.device}.")

    def on_validation_batch_start(  # type: ignore # pragma: no cover
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: ty.Any,
        batch_idx: int,
        *args: ty.Any,
    ) -> None:
        self._batch_type_info(batch, batch_idx, "Validation")

    def on_validation_batch_end(  # type: ignore # pragma: no cover
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: ty.Any,
        batch: ty.Any,
        batch_idx: int,
        *args: ty.Any,
    ) -> None:
        self._log(f"On validation batch end. Got the following outputs: {outputs}")

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        *args: ty.Any,
    ) -> None:
        try:
            loaders = trainer.train_dataloader.loaders  # type: ignore
            loaders_info = f"The training dataloader(s) is/are: {loaders}"
        except Exception:  # pylint: disable=broad-except
            loaders_info = ""
        self._log(f"Training epoch {trainer.current_epoch} is starting. {loaders_info}")

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        *args: ty.Any,
    ) -> None:
        self._log(f"Training epoch {trainer.current_epoch} is over.")

    def on_validation_epoch_start(  # pragma: no cover
        self,
        trainer: pl.Trainer,
        *args: ty.Any,
    ) -> None:
        self._log(f"Validation epoch {trainer.current_epoch} is starting.")

    def on_validation_epoch_end(  # pragma: no cover
        self,
        trainer: pl.Trainer,
        *args: ty.Any,
    ) -> None:
        epoch = trainer.current_epoch
        self._log(f"Validation epoch {epoch} is over.")

    def on_test_epoch_start(  # pragma: no cover
        self,
        trainer: pl.Trainer,
        *args: ty.Any,
    ) -> None:
        epoch = trainer.current_epoch
        self._log(f"Test epoch {epoch} is starting.")

    def on_test_epoch_end(  # pragma: no cover
        self,
        trainer: pl.Trainer,
        *args: ty.Any,
    ) -> None:
        epoch = trainer.current_epoch
        self._log(f"Test epoch {epoch} is over.")

    def _batch_type_info(
        self,
        batch: ty.Any,
        batch_idx: int,
        stage: str,
    ) -> None:
        try:
            type_info = f" batch content={type(batch[0])}"
        except Exception:
            type_info = ""
        self._log(f"{stage} batch: idx={batch_idx}; batch type={type(batch)};{type_info}")

    def _get_optim_info(self, optimizer: torch.optim.Optimizer) -> str:
        try:
            msg = "Current optimizer: "
            for param_group in optimizer.param_groups:
                msg += f'{type(optimizer).__name__}({param_group["name"]}) '
        except Exception as ex:
            msg = f"Error in retrieving optimizer's information: {ex}"
        return msg
