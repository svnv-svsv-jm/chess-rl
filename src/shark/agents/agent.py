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
            env (gym.Env):
                Game environment.

            replay_buffer:
                Replay buffer for storing experiences.
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
            net (nn.Module):
                DQN network.

            epsilon (float):
                Value to determine likelihood of taking a random action

            device (torch.device | str):
                Current device.

        Returns:
            action (int).
        """
        # Check if we take a random action
        if np.random.random() < epsilon:
            action: int = self.env.action_space.sample()
        else:
            state = torch.tensor([self.state]).to(device)
            q_values = net(state)
            _, action_ = torch.max(q_values, dim=1)
            action = int(action_.item())
        # Return action
        return action

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> ty.Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net (nn.Module):
                DQN network.

            epsilon (float):
                Value to determine likelihood of taking a random action

            device (torch.device | str):
                Current device.

        Returns:
            reward (float)

            done (bool)
        """
        # Get action from network
        action = self.get_action(net, epsilon, device)
        # Take a step into the game
        next_state, reward, done, _, _ = self.env.step(action)
        next_state = _fix_fucked_up_state(next_state)
        # Create experience data and update replay buffer
        exp = Experience(
            state=self.state,
            action=action,
            reward=float(reward),
            done=done,
            next_state=next_state,
        )
        self.replay_buffer.append(exp)
        # Update state and reset if done
        self.state = next_state
        if done:
            self.reset()
        # Return reward and done state
        return float(reward), done


def _fix_fucked_up_state(next_state: np.ndarray) -> np.ndarray:
    """Fix broken envs."""
    if not isinstance(next_state, np.ndarray):
        if isinstance(next_state, (list, tuple)):  # pragma: no cover
            next_state = next_state[0]  # pragma: no cover
        if not isinstance(next_state, np.ndarray):  # pragma: no cover
            logger.warning(f"Unsupported type {type(next_state)}, expected {np.ndarray}.")
    return next_state
