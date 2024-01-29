__all__ = ["ReplayBuffer"]

from loguru import logger
import typing as ty
import collections
import numpy as np

from shark.types import Experience


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them."""

    def __init__(self, capacity: int) -> None:
        """
        Args:
            capacity (int):
                Size of the buffer.
        """
        self.buffer: collections.deque[Experience] = collections.deque(maxlen=capacity)

    def __len__(self) -> int:
        length: int = len(self.buffer)
        return length

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.
        Args:
            experience (`Experience`):
                Tuple `(state, action, reward, done, new_state)`.
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> ty.Tuple:
        """Sample from the buffer."""
        # Get random indices with replacement based on batch size
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        # Create empty lists for each element in experience tuple
        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []
        # Loop through each index and add experience at that index to each list
        for idx in indices:
            experience: Experience = self.buffer[idx]
            # State
            state = experience.state
            if isinstance(state, tuple):
                state = state[0]
            logger.trace(f"State ({type(state)}): {state}")
            states.append(state)
            # Action
            action = experience.action
            logger.trace(f"Action ({type(action)}): {action}")
            actions.append(action)
            # Reward
            reward = experience.reward
            logger.trace(f"Reward ({type(reward)}): {reward}")
            rewards.append(reward)
            # Done
            done = experience.done
            logger.trace(f"Done ({type(done)}): {done}")
            dones.append(done)
            # Next State
            next_state = experience.next_state
            logger.trace(f"Next State ({type(next_state)}): {next_state}")
            next_states.append(next_state)
        # Return
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype="bool"),
            np.array(next_states),
        )
