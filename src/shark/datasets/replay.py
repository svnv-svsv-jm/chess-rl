__all__ = ["ReplayBufferDataset"]

import typing as ty
from torch.utils.data import IterableDataset

from shark.buffers import ReplayBuffer


class ReplayBufferDataset(IterableDataset):
    """Iterable Dataset containing the `ReplayBuffer` which will be updated with new experiences during training."""

    def __init__(
        self,
        buffer: ReplayBuffer,
        sample_size: int = 200,
    ) -> None:
        """
        Args:
            buffer (ReplayBuffer):
                Replay buffer.
            sample_size (int, optional):
                Number of experiences to sample at a time. Defaults to 200.
        """
        self.buffer = buffer
        self.sample_size = sample_size

    def __getitem__(self, i: int) -> ty.Any:
        raise NotImplementedError(f"{self.__class__.__name__} is not indexable.")

    def __iter__(self) -> ty.Iterator[ty.Tuple]:
        """Sample from `ReplayBuffer` and yield experiences."""
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i, done in enumerate(dones):
            yield states[i], actions[i], rewards[i], done, new_states[i]
