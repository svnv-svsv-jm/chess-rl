__all__ = ["Experience"]

import typing as ty
from pydantic import BaseModel
import numpy as np

NUMPY_ARRAY_TYPE = np.ndarray


class Experience(BaseModel):
    """Experience."""

    class Config:
        """Config."""

        arbitrary_types_allowed = True

    state: NUMPY_ARRAY_TYPE
    action: int
    reward: float
    done: bool
    next_state: NUMPY_ARRAY_TYPE
