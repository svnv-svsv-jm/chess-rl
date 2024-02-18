__all__ = ["DQN"]

from loguru import logger
import typing as ty
import lightning.pytorch as pl
import gymnasium as gym
import copy
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam
from torch import Tensor
from torch import nn
from torchrl.modules import MLP

from shark.buffers import ReplayBuffer
from shark.agents import Agent
from shark.datasets import ReplayBufferDataset


class DQN(pl.LightningModule):
    """Basic DQN Model"""

    def __init__(
        self,
        batch_size: int = 16,
        lr: float = 1e-2,
        env: str = "CartPole-v0",
        gamma: float = 0.99,
        sync_rate: int = 10,
        replay_size: int = 1000,
        warm_start_size: int = 1000,
        eps_last_frame: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        episode_length: int = 500,
        warm_start_steps: int = 1000,
        hidden_dims: ty.Sequence[int] = [128],
    ) -> None:
        """
        Args:
            batch_size (int):
                Size of the batches.
            lr (float):
                Learning rate.
            env (str):
                Gym environment tag.
            gamma (float):
                Reward discount factor.
            sync_rate (int):
                Number of steps to take before updating the target network.
            replay_size (int):
                Capacity of the replay buffer
            warm_start_size (int):
                Number of samples to fill our buffer at the start of training.
            eps_last_frame (int):
                What frame should epsilon stop decaying.
            eps_start (float):
                Starting value of epsilon
            eps_end (float):
                Final value of epsilon.
            episode_length (int):
                Max length of an episode.
            warm_start_steps (int):
                Max episode reward in the environment.
            hidden_dims (Sequence[int]):
                Hidden layers for MLP network.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.env_name = env
        self.env: gym.Env = gym.make(env)
        self.obs_size = int(self.env.observation_space.shape[0])  # type: ignore
        self.n_actions = int(self.env.action_space.n)  # type: ignore

        self.net = MLP(out_features=self.n_actions, num_cells=hidden_dims)
        self.target_net = MLP(out_features=self.n_actions, num_cells=hidden_dims)

        self.buffer = ReplayBuffer(replay_size)
        self.agent = Agent(self.env, self.buffer)

        self.criterion = nn.MSELoss()

        self.total_reward = 0.0
        self.episode_reward = 0.0

        self.populate(warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with experiences

        Args:
            steps: number of random steps to populate the buffer with
        """
        for _ in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output

        Args:
            x: environment state.

        Returns:
            q values
        """
        output: Tensor = self.net(x)
        return output

    def dqn_mse_loss(self, batch: ty.Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        """Calculates the MSE loss using a mini batch from the replay buffer.

        Args:
            batch (`Sequence[Tensor]`):
                Current batch of replay data.

        Returns:
            Tensor:
                Loss value.
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        gamma: float = self.hparams["gamma"]
        expected_state_action_values = next_state_values * gamma + rewards

        loss: Tensor = self.criterion(state_action_values, expected_state_action_values)
        return loss

    def training_step(
        self,
        batch: ty.Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss based on the mini-batch recieved.

        Args:
            batch (ty.Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]):
                Current mini batch of replay data
            batch_idx (int):
                Batch number.

        Returns:
            Training loss.
        """
        device = self.get_device(batch)
        eps_end: float = self.hparams["eps_end"]
        eps_start: float = self.hparams["eps_start"]
        eps_last_frame: float = self.hparams["eps_last_frame"]
        epsilon = max(eps_end, eps_start - self.global_step + 1 / eps_last_frame)

        # step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        sync_rate: int = self.hparams["sync_rate"]
        if self.global_step % sync_rate == 0:
            try:
                self.target_net.load_state_dict(self.net.state_dict())
            except RuntimeError as ex:
                logger.warning(ex)  # pragma: no cover
                self.target_net = copy.deepcopy(self.net)  # pragma: no cover

        self.log_dict(
            {
                "reward": reward,
                "train_loss": loss,
            }
        )
        self.log("total_reward", self.total_reward, prog_bar=True)
        self.log("steps", self.global_step, logger=False, prog_bar=True)

        return loss

    def configure_optimizers(self) -> ty.List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def train_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        episode_length: int = self.hparams["episode_length"]
        batch_size: int = self.hparams["batch_size"]
        dataset = ReplayBufferDataset(self.buffer, episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
        )
        return dataloader

    def get_device(self, batch: ty.Sequence[Tensor]) -> torch.device:
        """Retrieve device currently being used by mini-batch."""
        t: Tensor = batch[0]
        device = t.device
        return device
