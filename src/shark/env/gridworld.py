from typing import Any, SupportsFloat, TypeVar
import gymnasium as gym
from gymnasium import spaces
import numpy as np

STATE = TypeVar("STATE")
ACTION = TypeVar("ACTION")


class GridWorldEnv(gym.Env):
    """Grid world."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: str = None, size: int = 5) -> None:
        """
        Args:
            render_mode (str, optional): _description_. Defaults to None.
            size (int, optional): _description_. Defaults to 5.
        """
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    0,
                    size - 1,
                    shape=(2,),
                    dtype=int,  # type: ignore
                ),
                "target": spaces.Box(
                    0,
                    size - 1,
                    shape=(2,),
                    dtype=int,  # type: ignore
                ),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def reset(
        self,
        *,
        seed: int = None,
        options: dict[str, Any] = None,
    ) -> tuple[STATE, dict[str, Any]]:
        return super().reset(seed=seed, options=options)

    def step(self, action: ACTION) -> tuple[ACTION, SupportsFloat, bool, bool, dict[str, Any]]:
        return super().step(action)

    def render(self) -> None:
        super().render()

    def close(self) -> None:
        pass
