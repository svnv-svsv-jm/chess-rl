__all__ = ["Agent"]

from loguru import logger
import typing as ty
import gymnasium as gym
import numpy as np
import torch
from torch import nn

from shark.types import Experience
from shark.buffers import ReplayBuffer


class Agent:
    """Base Agent class handling the interaction with the environment."""

    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.state: np.ndarray
        self.reset()

    def reset(self) -> None:
        """Resents the environment and updates the state."""
        state, _ = self.env.reset()
        self.state = _fix_fucked_up_state(state)

    def get_action(
        self,
        net: nn.Module,
        epsilon: float,
        device: ty.Union[torch.device, str] = "cpu",
    ) -> int:
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        if np.random.random() < epsilon:
            action: int = self.env.action_space.sample()
        else:
            state = torch.tensor([self.state]).to(device)
            q_values = net(state)
            _, action_ = torch.max(q_values, dim=1)
            action = int(action_.item())

        return action

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float = 0.0,
        device: ty.Union[torch.device, str] = "cpu",
    ) -> ty.Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """
        action = self.get_action(net, epsilon, device)

        next_state, reward, done, _, _ = self.env.step(action)
        next_state = _fix_fucked_up_state(next_state)

        exp = Experience(
            state=self.state,
            action=action,
            reward=float(reward),
            done=done,
            next_state=next_state,
        )

        self.replay_buffer.append(exp)

        self.state = next_state
        if done:
            self.reset()
        return float(reward), done


def _fix_fucked_up_state(next_state: np.ndarray) -> np.ndarray:
    """Fix broken envs."""
    if not isinstance(next_state, np.ndarray):
        if isinstance(next_state, (list, tuple)):
            next_state = next_state[0]
        if not isinstance(next_state, np.ndarray):
            logger.warning(f"Unsupported type {type(next_state)}, expected {np.ndarray}.")
    return next_state
