import collections
from typing import Deque, List

import numpy as np
from stable_baselines3.common import callbacks, vec_env

from imitation.rewards import common
from imitation.data.types import Trajectory

import gym
from gym import spaces


class FlatObsOnlyWrapper(gym.core.ObservationWrapper):
    """
    Flatten the image as the only observation output, no language/mission.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']

    def observation(self, obs):
        return obs['image'].flatten()


class EncoderWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env, encoder):
        super().__init__(env)

        self.encoder = encoder
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(encoder.feature_dim,)
        )

    def observation(self, obs):
        return self.encoder.encode(obs)

class WrappedRewardCallback(callbacks.BaseCallback):
    """Logs mean wrapped reward as part of RL (or other) training."""

    def __init__(self, episode_rewards: Deque[float], *args, **kwargs):
        self.episode_rewards = episode_rewards
        super().__init__(self, *args, **kwargs)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        if len(self.episode_rewards) == 0:
            return
        mean = sum(self.episode_rewards) / len(self.episode_rewards)
        self.logger.record("rollout/ep_rew_wrapped_mean", mean)
