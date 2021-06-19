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

class RewardVecEnvWrapperRNN(vec_env.VecEnvWrapper):
    def __init__(
        self, venv: vec_env.VecEnv, reward_fn: common.RewardFn, ep_history: int = 100, bac_reward_flag: bool = False
    ):
        """Uses a provided reward_fn to replace the reward function returned by `step()`.

        Automatically resets the inner VecEnv upon initialization. A tricky part
        about this class is keeping track of the most recent observation from each
        environment.

        Will also include the previous reward given by the inner VecEnv in the
        returned info dict under the `wrapped_env_rew` key.

        

        Args:
            venv: The VecEnv to wrap.
            reward_fn: A function that wraps takes in vectorized transitions
                (obs, act, next_obs) a vector of episode timesteps, and returns a
                vector of rewards.
            ep_history: The number of episode rewards to retain for computing
                mean reward.
            bac_reward_flag: Whether to use new_rews(False) or old_rews-new_rews(True).
        """
        assert not isinstance(venv, RewardVecEnvWrapperRNN)
        super().__init__(venv)
        self.episode_rewards = collections.deque(maxlen=ep_history)
        self._cumulative_rew = np.zeros((venv.num_envs,))
        self.reward_fn = reward_fn
        
        self.obs_list = []
        self.act_list = []
        self.bac_reward_flag = bac_reward_flag 
        
        self.reset()

    def make_log_callback(self) -> WrappedRewardCallback:
        """Creates `WrappedRewardCallback` connected to this `RewardVecEnvWrapper`."""
        return WrappedRewardCallback(self.episode_rewards)

    @property
    def envs(self):
        return self.venv.envs

    def reset(self):
        self._old_obs = self.venv.reset()
        self.obs_list = [np.squeeze(self._old_obs,axis=0)]
        self.act_list = []
        return self._old_obs

    def step_async(self, actions):
        self._actions = actions
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, old_rews, dones, infos = self.venv.step_wait()

        # The vecenvs automatically reset the underlying environments once they
        # encounter a `done`, in which case the last observation corresponding to
        # the `done` is dropped. We're going to pull it back out of the info dict!
        obs_fixed = []
        for single_obs, single_done, single_infos in zip(obs, dones, infos):
            if single_done:
                single_obs = single_infos["terminal_observation"]

            obs_fixed.append(single_obs)
        obs_fixed = np.stack(obs_fixed)

        self.obs_list.append(np.squeeze(obs_fixed,axis=0)) #LIST OF OBS
        self.act_list.append(np.squeeze(self._actions,axis=0)) #LIST OF ACT
        print(np.array(self.obs_list).shape, np.array(self.act_list).shape)


        rews = self.reward_fn(
            Trajectory(
            obs=np.array(self.obs_list),
            acts=np.array(self.act_list),
            infos=np.array([{} for i in self.act_list]),
            )
        ).item()

        if self.bac_reward_flag:
            rews = old_rews - rews
        
        # assert len(rews) == len(obs), "must return one rew for each env"
        done_mask = np.asarray(dones, dtype="bool").reshape((len(dones),))

        # Update statistics
        self._cumulative_rew += rews
        for single_done, single_ep_rew in zip(dones, self._cumulative_rew):
            if single_done:
                self.episode_rewards.append(single_ep_rew)
        self._cumulative_rew[done_mask] = 0

        # we can just use obs instead of obs_fixed because on the next iteration
        # after a reset we DO want to access the first observation of the new
        # trajectory, not the last observation of the old trajectory
        self._old_obs = obs
        for info_dict, old_rew in zip(infos, old_rews):
            info_dict["wrapped_env_rew"] = old_rew
        return obs, rews, dones, infos

class RewardVecEnvWrapper(vec_env.VecEnvWrapper):
    def __init__(
        self, venv: vec_env.VecEnv, reward_fn: common.RewardFn, ep_history: int = 100, bac_reward_flag: bool = False
    ):
        """Uses a provided reward_fn to replace the reward function returned by `step()`.

        Automatically resets the inner VecEnv upon initialization. A tricky part
        about this class is keeping track of the most recent observation from each
        environment.

        Will also include the previous reward given by the inner VecEnv in the
        returned info dict under the `wrapped_env_rew` key.

        

        Args:
            venv: The VecEnv to wrap.
            reward_fn: A function that wraps takes in vectorized transitions
                (obs, act, next_obs) a vector of episode timesteps, and returns a
                vector of rewards.
            ep_history: The number of episode rewards to retain for computing
                mean reward.
            bac_reward_flag: Whether to use new_rews(False) or old_rews-new_rews(True).
        """
        assert not isinstance(venv, RewardVecEnvWrapper)
        super().__init__(venv)
        self.episode_rewards = collections.deque(maxlen=ep_history)
        self._cumulative_rew = np.zeros((venv.num_envs,))
        self.reward_fn = reward_fn

        self.bac_reward_flag = bac_reward_flag

        self.reset()

    def make_log_callback(self) -> WrappedRewardCallback:
        """Creates `WrappedRewardCallback` connected to this `RewardVecEnvWrapper`."""
        return WrappedRewardCallback(self.episode_rewards)

    @property
    def envs(self):
        return self.venv.envs

    def reset(self):
        self._old_obs = self.venv.reset()
        return self._old_obs

    def step_async(self, actions):
        self._actions = actions
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, old_rews, dones, infos = self.venv.step_wait()

        # The vecenvs automatically reset the underlying environments once they
        # encounter a `done`, in which case the last observation corresponding to
        # the `done` is dropped. We're going to pull it back out of the info dict!
        obs_fixed = []
        for single_obs, single_done, single_infos in zip(obs, dones, infos):
            if single_done:
                single_obs = single_infos["terminal_observation"]

            obs_fixed.append(single_obs)
        obs_fixed = np.stack(obs_fixed)

        rews = self.reward_fn(self._old_obs, self._actions, obs_fixed, np.array(dones))
        
        if self.bac_reward_flag:
            rews = old_rews - rews
        
        assert len(rews) == len(obs), "must return one rew for each env"
        done_mask = np.asarray(dones, dtype="bool").reshape((len(dones),))

        # Update statistics
        self._cumulative_rew += rews
        for single_done, single_ep_rew in zip(dones, self._cumulative_rew):
            if single_done:
                self.episode_rewards.append(single_ep_rew)
        self._cumulative_rew[done_mask] = 0

        # we can just use obs instead of obs_fixed because on the next iteration
        # after a reset we DO want to access the first observation of the new
        # trajectory, not the last observation of the old trajectory
        self._old_obs = obs
        for info_dict, old_rew in zip(infos, old_rews):
            info_dict["wrapped_env_rew"] = old_rew
        return obs, rews, dones, infos
