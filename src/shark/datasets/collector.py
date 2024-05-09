__all__ = ["CollectorDataset", "make_collector", "make_chess_collector"]

from loguru import logger
import typing as ty
import torch
from torch.utils.data import IterableDataset
from torchrl.data import MultiStep
from torchrl.collectors import MultiSyncDataCollector, MultiaSyncDataCollector, DataCollectorBase
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import EnvBase, ExplorationType
from tensordict.nn import TensorDictModule
from tensordict import TensorDict

from shark.env import make_chess_env
from shark.utils import find_device
from .patch import SyncDataCollector


def make_collector(
    create_env_fn: ty.Callable[..., EnvBase],
    actor: TensorDictModule,
    num_collectors: int,
    device: torch.device,
    collector_type: str,
    # num_workers: int,
    # parallel: bool,
    **kwargs: ty.Any,
) -> DataCollectorBase:
    """Create data collector."""
    create_env_fn_ = (
        create_env_fn(**kwargs)
        if num_collectors == 1
        else [create_env_fn(**kwargs)] * num_collectors
    )
    params = dict(
        create_env_fn=create_env_fn_,
        policy=actor,
        frames_per_batch=1,
        total_frames=10,
        # this is the default behaviour: the collector runs in ``"random"`` (or explorative) mode
        exploration_type=ExplorationType.RANDOM,
        # We set the all the devices to be identical. Below is an example of
        # heterogeneous devices
        device=device,
        storing_device=device,
        split_trajs=False,
        postproc=MultiStep(gamma=0.98, n_steps=5),
    )
    collector_type = collector_type.lower()
    if collector_type in ["sync"]:
        collector = SyncDataCollector(**params)
    elif collector_type in ["multi", "multi-sync", "multisync"]:
        collector = MultiSyncDataCollector(**params)
    elif collector_type in ["multiasync", "multi-async"]:
        collector = MultiaSyncDataCollector(**params)
    else:
        raise ValueError(f"Invalid collector type {collector_type}.")
    return collector


def make_chess_collector(
    engine_executable: str,
    num_workers: int,
    parallel: bool,
    **kwargs: ty.Any,
) -> DataCollectorBase:
    """Create data collector."""
    return make_collector(
        create_env_fn=make_chess_env,
        engine_executable=engine_executable,
        num_workers=num_workers,
        parallel=parallel,
        **kwargs,
    )


class CollectorDataset(IterableDataset):
    """Iterable Dataset containing the `ReplayBuffer` which will be updated with new experiences during training, and the `SyncDataCollector | MultiSyncDataCollector | MultiaSyncDataCollector`."""

    def __init__(
        self,
        env: EnvBase | ty.Sequence[ty.Callable[..., EnvBase]],
        policy_module: TensorDictModule,
        frames_per_batch: int,
        total_frames: int,
        device: torch.device = find_device(),
        split_trajs: bool = False,
        batch_size: int = 1,
        init_random_frames: int = 1,
        collector_type: str = "sync",
        postproc: ty.Optional[torch.nn.Module] = MultiStep(gamma=0.98, n_steps=5),
        reshape: bool = True,
        **kwargs: ty.Any,
    ) -> None:
        # Attributes
        self.batch_size = batch_size
        self.device = device
        self.env = env
        self.policy_module = policy_module
        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames
        self.collector_type = collector_type
        self.reshape = reshape
        # Get num envs
        num_collectors = 1
        if isinstance(self.env, ty.Sequence):
            num_collectors = len(self.env)
        self.num_collectors = num_collectors
        # Collector's params
        params = dict(
            create_env_fn=self.env,
            policy=self.policy_module,
            frames_per_batch=self.frames_per_batch,
            total_frames=self.total_frames,
            device=self.device,
            storing_device=self.device,
            split_trajs=split_trajs,
            init_random_frames=init_random_frames,
            postproc=postproc,
        )
        params.update(kwargs)
        # Collector
        if collector_type in ["sync"] or self.num_collectors < 2:
            self.collector = SyncDataCollector(**params)
        elif collector_type in ["multi", "multi-sync", "multisync"]:
            self.collector = MultiSyncDataCollector(**params)
        elif collector_type in ["multiasync", "multi-async"]:
            self.collector = MultiaSyncDataCollector(**params)
        else:
            raise ValueError(f"Invalid collector type {collector_type}.")
        # ReplayBuffer
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(frames_per_batch),
            sampler=SamplerWithoutReplacement(),
            batch_size=self.batch_size,
        )

    @property
    def length(self) -> int:
        """Size of dataset."""
        L = len(self.replay_buffer)
        if self.total_frames > L:
            return self.total_frames
        return L

    def __iter__(self) -> ty.Iterator[TensorDict]:
        """Yield experiences from `SyncDataCollector` and store them in `ReplayBuffer`."""
        i = 0
        for i, tensordict_data in enumerate(self.collector):
            assert isinstance(tensordict_data, TensorDict)
            if self.reshape:
                data_view: TensorDict = tensordict_data.reshape(-1)
            else:
                data_view = tensordict_data
            logger.trace(f"Collecting ({i}): {data_view.shape} | {data_view.device}")
            self.replay_buffer.extend(data_view.cpu())
            yield tensordict_data.to(self.device)

    def sample(self, *args: ty.Any, **kwargs: ty.Any) -> TensorDict:
        """Sample from `ReplayBuffer`."""
        data: TensorDict = self.replay_buffer.sample(*args, **kwargs)
        return data.to(self.device)
