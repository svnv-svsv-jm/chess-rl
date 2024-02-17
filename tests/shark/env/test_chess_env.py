import pytest
from loguru import logger
import typing as ty
import sys, os

from tensordict import TensorDict
from torchrl.envs.utils import check_env_specs
from torchrl.collectors import RandomPolicy, SyncDataCollector

from shark.env._custom import _CustomEnv
from shark.env import ChessEnv
from shark.utils import find_device


@pytest.mark.parametrize("custom, from_engine", [(True, None), (False, False), (False, True)])
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
        )
    env.set_seed(0)
    policy = RandomPolicy(env.action_spec)
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=2,
        total_frames=10,
        device=device,
        reset_at_each_iter=True,
    )
    for i, td in enumerate(collector):
        assert isinstance(td, TensorDict)
        if i > 2:
            break
    logger.success(f"Passed for {env.__class__.__name__}")


def test_chess_env(engine_executable: str) -> None:
    """Test we can initialize the chess environment."""
    # Create env
    env = ChessEnv(
        engine_path=engine_executable,
        play_as="black",
        device=find_device(),
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


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
