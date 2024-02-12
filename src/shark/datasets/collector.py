__all__ = ["CollectorDataset"]

from loguru import logger
import typing as ty
import torch
from torch.utils.data import IterableDataset, Dataset
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import EnvBase
from tensordict.nn import TensorDictModule
from tensordict import TensorDict

from shark.utils import find_device


class CollectorDataset(IterableDataset):
    """Iterable Dataset containing the `ReplayBuffer` which will be updated with new experiences during training, and the `SyncDataCollector`."""

    def __init__(
        self,
        env: EnvBase,
        policy_module: TensorDictModule,
        frames_per_batch: int,
        total_frames: int,
        device: torch.device = find_device(),
        split_trajs: bool = False,
        batch_size: int = 1,
        init_random_frames: int = 1,
    ) -> None:
        # Attributes
        self.batch_size = batch_size
        self.device = device
        self.env = env
        self.policy_module = policy_module
        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames
        # Collector
        self.collector = SyncDataCollector(
            self.env,
            self.policy_module,
            frames_per_batch=self.frames_per_batch,
            total_frames=self.total_frames,
            device=self.device,
            storing_device=self.device,
            split_trajs=split_trajs,
            init_random_frames=init_random_frames,
        )
        # ReplayBuffer
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(frames_per_batch),
            sampler=SamplerWithoutReplacement(),
            batch_size=self.batch_size,
        )
        # States
        self.length: ty.Optional[int] = None

    # def __len__(self) -> int:
    #     """Return the number of experiences in the `ReplayBuffer`."""
    #     if self.length is not None:
    #         return self.length
    #     L = len(self.replay_buffer)
    #     if self.total_frames > L:
    #         return self.total_frames
    #     return L

    def __iter__(self) -> ty.Iterator[TensorDict]:
        """Yield experiences from `SyncDataCollector` and store them in `ReplayBuffer`."""
        i = 0
        for i, tensordict_data in enumerate(self.collector):
            logger.trace(f"Collecting {i}")
            assert isinstance(tensordict_data, TensorDict)
            data_view: TensorDict = tensordict_data.reshape(-1)
            self.replay_buffer.extend(data_view.cpu())
            yield tensordict_data.to(self.device)
        self.length = i

    # def __getitem__(self, idx: int = None, **kwargs: ty.Any) -> TensorDict:
    #     """Sample from `ReplayBuffer`."""
    #     return self.sample(**kwargs)

    def sample(self, **kwargs: ty.Any) -> TensorDict:
        """Sample from `ReplayBuffer`."""
        data: TensorDict = self.replay_buffer.sample(**kwargs)
        return data.to(self.device)
