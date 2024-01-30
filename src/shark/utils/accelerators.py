__all__ = ["find_device"]

import typing as ty

import torch
from lightning.pytorch.accelerators.cuda import CUDAAccelerator
from lightning.pytorch.accelerators.mps import MPSAccelerator


def find_device(accelerator: ty.Union[torch.device, str] = "auto") -> torch.device:
    """Automatically finds system's device for PyTorch."""
    if isinstance(accelerator, torch.device):
        return accelerator
    device = _choose_auto_accelerator(accelerator)
    assert device in ("cpu", "mps", "cuda")
    return torch.device(device)


def _choose_auto_accelerator(accelerator_flag: str) -> str:
    """Choose the accelerator type (str) based on availability when ``accelerator='auto'``."""
    accelerator_flag = accelerator_flag.lower()
    assert accelerator_flag in ("auto", "cpu", "mps", "cuda")
    try:
        if accelerator_flag == "auto":
            if MPSAccelerator.is_available():
                return "mps"
            if CUDAAccelerator.is_available():
                return "cuda"
        return "cpu"
    except Exception:
        return "cpu"
