import math
from types import SimpleNamespace
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from carla_MO import MOCarlaWrapper


class CarlaMOGymEnv(gym.Env):
    """Gymnasium adapter for the CARLA multi-objective wrapper."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        town: str = "Town04",
        episode_length: int = 200,
        delta_seconds: float = 0.05,
        action_repeat: int = 4,
    ) -> None:
        super().__init__()
        self.base = MOCarlaWrapper(
            host=host,
            port=port,
            town=town,
            episode_length=episode_length,
            delta_seconds=delta_seconds,
            action_repeat=action_repeat,
        )

        # Kinematics (15, 7) + LiDAR (64,) -> flattened 169-dim observation.
        obs_dim = 15 * 7 + 64
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        # Continuous action [throttle, steer] where throttle can be negative
        # in base.step() to denote braking.
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        # MORL-Baselines expects reward_space on env.unwrapped.
        self.reward_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 0.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.spec = SimpleNamespace(id="carla-mo-v0")

    @staticmethod
    def _flatten_obs(obs: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        kinematics_obs, lidar_obs = obs
        return np.concatenate([kinematics_obs.reshape(-1), lidar_obs.reshape(-1)]).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        del seed, options
        obs, info = self.base.reset()
        return self._flatten_obs(obs), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        obs, vec_reward, terminated, truncated, info = self.base.step(action)
        return self._flatten_obs(obs), vec_reward.astype(np.float32), terminated, truncated, info

    def close(self):
        if hasattr(self, "base") and self.base is not None:
            self.base.close()

