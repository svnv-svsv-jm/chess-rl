import pytest
from loguru import logger
import typing as ty
import sys, os

from tensordict import TensorDict
from torchrl.envs import EnvCreator, check_env_specs
from torchrl.collectors import RandomPolicy, SyncDataCollector

from shark.env._custom import _CustomEnv
from shark.env import ChessEnv
from shark.utils import find_device
from shark.utils.patch import ParallelEnv


@pytest.mark.parametrize("num_envs", [3])
def test_parallen_env(engine_executable: str, num_envs: int) -> None:
    """Test usage of `ChessEnv` in `ParallelEnv`."""
    make_env = EnvCreator(
        lambda: ChessEnv(
            engine_path=engine_executable,
            probability_move_is_random=0.5,
        )
    )
    # Makes identical copies of the env, runs them on dedicated processes
    env = ParallelEnv(num_envs, make_env)
    check_env_specs(env)
    rollout = env.rollout(2)
    logger.info(f"Rollout: {rollout}")
    assert len(rollout) == num_envs
    # Test integration with collector
    policy = RandomPolicy(env.action_spec)
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=2,
        total_frames=10,
        device="cpu",
        reset_at_each_iter=True,
    )
    for i, td in enumerate(collector):
        assert isinstance(td, TensorDict)
        if i > 1:
            break
    logger.success(f"Passed for {env.__class__.__name__}")


@pytest.mark.parametrize("custom, from_engine", [(True, True), (False, False), (False, True)])
def test_use_env_in_collector(engine_executable: str, custom: bool, from_engine: bool) -> None:
    """Test `SyncDataCollector` on env."""
    device = find_device()
    if custom:
        env = _CustomEnv()
    else:
        env = ChessEnv(
            engine_executable,
            device=device,
            lose_on_illegal_move=False,
            from_engine=from_engine,
            probability_move_is_random=0.5,
        )
    env.set_seed(0)
    policy = RandomPolicy(env.action_spec)
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=2,
        total_frames=10,
        device=device,
        reset_at_each_iter=False,
    )
    for i, td in enumerate(collector):
        assert isinstance(td, TensorDict)
        if i > 1:
            break
    logger.success(f"Passed for {env.__class__.__name__}")


@pytest.mark.parametrize("use_one_hot", [True])
def test_chess_env(engine_executable: str, use_one_hot: bool) -> None:
    """Test we can initialize the chess environment."""
    # Create env
    env = ChessEnv(
        engine_path=engine_executable,
        play_as="black",
        device=find_device(),
        use_one_hot=use_one_hot,
        probability_move_is_random=0.5,
    )
    # Reset
    state = env.reset()
    logger.info(f"Reset state: {state}")
    # Step
    action = env.sample()
    out = env.step(action)
    logger.info(f"Tensordict: {out}")
    # Rollout
    td = env.rollout(3)
    logger.info(f"Rollout: {td}")
    # Sanity check
    check_env_specs(env)
    # Sample
    env.sample(from_engine=False)
    env.sample(from_engine=True)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
