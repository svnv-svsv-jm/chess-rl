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

_ENV_FN_TYPE = ty.Callable[..., EnvBase]


def make_collector(
    create_env_fn: EnvBase | _ENV_FN_TYPE | ty.Sequence[_ENV_FN_TYPE],
    actor: TensorDictModule,
    device: int | str | torch.device,
    collector_type: str,
    num_collectors: int = 1,
    frames_per_batch: int = 1,
    total_frames: int = 10,
    split_trajs: bool = False,
    postproc: ty.Optional[torch.nn.Module] = MultiStep(gamma=0.98, n_steps=5),
    exploration_type: ExplorationType = ExplorationType.RANDOM,
    init_random_frames: int = None,
    env_fn_params: dict = None,
    **kwargs: ty.Any,
) -> DataCollectorBase:
    """Create data collector."""
    # Env params
    env_fn_params = {} if env_fn_params is None else env_fn_params.copy()
    # Env
    if isinstance(create_env_fn, EnvBase):
        create_env_fn_ = create_env_fn(**env_fn_params)
    elif callable(create_env_fn):
        create_env_fn_ = (
            create_env_fn(**env_fn_params)
            if num_collectors == 1
            else [create_env_fn(**env_fn_params)] * num_collectors
        )
    elif isinstance(create_env_fn, (list, tuple)):
        create_env_fn_ = [f(**env_fn_params) for f in create_env_fn_ if callable(f)]
    else:
        raise TypeError(
            f"`create_env_fn` must be {EnvBase} | {_ENV_FN_TYPE} | {ty.Sequence[_ENV_FN_TYPE]}."
        )
    # Collector params
    params = dict(
        create_env_fn=create_env_fn_,
        policy=actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        # this is the default behaviour: the collector runs in ``"random"`` (or explorative) mode
        exploration_type=exploration_type,
        # We set the all the devices to be identical. Below is an example of
        # heterogeneous devices
        device=device,
        storing_device=device,
        split_trajs=split_trajs,
        postproc=postproc,
        init_random_frames=init_random_frames,
    )
    params.update(kwargs)
    # Collector
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
    env_fn_params = dict(
        engine_executable=engine_executable,
        num_workers=num_workers,
        parallel=parallel,
    )
    return make_collector(
        create_env_fn=make_chess_env,
        env_fn_params=env_fn_params,
        **kwargs,
    )


class CollectorDataset(IterableDataset):
    """Iterable Dataset containing the `ReplayBuffer` which will be updated with new experiences during training, and the `SyncDataCollector | MultiSyncDataCollector | MultiaSyncDataCollector`."""

    def __init__(
        self,
        create_env_fn: EnvBase | _ENV_FN_TYPE | ty.Sequence[_ENV_FN_TYPE],
        policy_module: TensorDictModule,
        env_fn_params: dict = None,
        frames_per_batch: int = 1,
        total_frames: int = 10,
        device: int | str | torch.device = find_device(),
        split_trajs: bool = False,
        batch_size: int = 1,
        init_random_frames: int = None,
        collector_type: str = "sync",
        postproc: ty.Optional[torch.nn.Module] = MultiStep(gamma=0.98, n_steps=5),
        reshape: bool = True,
        exploration_type: ExplorationType = ExplorationType.RANDOM,
        **kwargs: ty.Any,
    ) -> None:
        """
        Args:
            create_env_fn (EnvBase | _ENV_FN_TYPE | ty.Sequence[_ENV_FN_TYPE]):
                A RL environment, or a callable that returns a RL environment, or a sequence of those.

            policy_module (TensorDictModule):
                The actor / agent module.

            env_fn_params (dict, optional):
                Keyword arguments for the environment function. Defaults to `None`.

            frames_per_batch (int):
                Frames per batch.

            total_frames (int):
                Total number of frames.

            device (int | str | torch.device, optional):
                PyTorch device. Defaults to `find_device()`, which looks for `"mps"`, `"cuda"`, `"cpu"` (in this order).

            split_trajs (bool, optional):
                Boolean indicating whether the resulting `TensorDict` should be split according to the trajectories.
                See :func:`~torchrl.collectors.utils.split_trajectories` for more information.
                Defaults to `False`.

            batch_size (int, optional):
                Batch size for :class:`torchrl.data.replay_buffers.ReplayBuffer`. Defaults to `1`.

            init_random_frames (int, optional):
                Number of frames for which the policy is ignored before it is called.
                This feature is mainly intended to be used in offline/model-based settings, where a
                batch of random trajectories can be used to initialize training.
                If provided, it will be rounded up to the closest multiple of frames_per_batch.
                Defaults to `None` (i.e. no random frames).

            collector_type (str, optional):
                Type of collector. Defaults to `"sync"`.

            postproc (ty.Optional[torch.nn.Module], optional):
                Post-processing callable. Defaults to `MultiStep(gamma=0.98, n_steps=5)`.

            reshape (bool, optional):
                Whether to reshape (`.reshape(-1)`) sampled batches from `ReplayBuffer`.
                Defaults to `True`.

            exploration_type (ExplorationType, optional):
                Interaction mode to be used when collecting data.
                Must be one of `torchrl.envs.utils.ExplorationType.RANDOM`,
                `torchrl.envs.utils.ExplorationType.MODE` or `torchrl.envs.utils.ExplorationType.MEAN`.
                Defaults to `ExplorationType.RANDOM`.
        """
        # Attributes
        self.batch_size = batch_size
        self.device = device
        self.create_env_fn = create_env_fn
        self.policy_module = policy_module
        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames
        self.collector_type = collector_type
        self.reshape = reshape
        # Get num envs
        num_collectors = 1
        if isinstance(self.create_env_fn, ty.Sequence):
            num_collectors = len(self.create_env_fn)
        self.num_collectors = num_collectors
        # Collector
        self.collector = make_collector(
            create_env_fn=self.create_env_fn,
            env_fn_params=env_fn_params,
            actor=self.policy_module,
            device=self.device,
            collector_type=self.collector_type,
            frames_per_batch=self.frames_per_batch,
            total_frames=self.total_frames,
            num_collectors=self.num_collectors,
            split_trajs=split_trajs,
            init_random_frames=init_random_frames,
            postproc=postproc,
            exploration_type=exploration_type,
            **kwargs,
        )
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
