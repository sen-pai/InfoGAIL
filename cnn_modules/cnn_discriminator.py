import gym
import numpy as np
import torch as th
from torch import nn

from imitation.util import networks

from stable_baselines3.common import preprocessing
from stable_baselines3.common.torch_layers import NatureCNN


class ActObsCNN(nn.Module):
    """CNN that takes an action and an image observation and produces a single
    output."""

    def __init__(
        self, action_space: gym.Space, observation_space: gym.Space, **mlp_kwargs
    ):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        self.cnn_feature_extractor = NatureCNN(observation_space, features_dim=256)

        self.in_size = (
            self.cnn_feature_extractor.features_dim
            + preprocessing.get_flattened_obs_dim(action_space)
        )

        self.mlp = networks.build_mlp(
            **{
                "in_size": self.in_size,
                "out_size": 1,
                "hid_sizes": (32, 32),
                **mlp_kwargs,
            }
        )

    def forward(self, obs: th.Tensor, acts: th.Tensor) -> th.Tensor:
        obs_features = self.cnn_feature_extractor(obs)
        cat_inputs = th.cat((obs_features, acts), dim=1)
        outputs = self.mlp(cat_inputs)

        return outputs.squeeze(1)

    def device(self) -> th.device:
        """Heuristic to determine which device this module is on."""
        first_param = next(self.parameters())
        return first_param.device


class ObsOnlyCNN(nn.Module):
    """CNN that uses only an image observation and produces a single
    output."""

    def __init__(
        self, action_space: gym.Space, observation_space: gym.Space, **mlp_kwargs
    ):
        super().__init__()

        self.cnn_feature_extractor = NatureCNN(observation_space, features_dim=512)

        in_size = self.cnn_feature_extractor.features_dim
        self.mlp = networks.build_mlp(
            **{"in_size": in_size, "out_size": 1, "hid_sizes": (32, 32), **mlp_kwargs}
        )

    def forward(self, obs: th.Tensor, acts: th.Tensor) -> th.Tensor:
        obs_features = self.cnn_feature_extractor(obs)
        outputs = self.mlp(obs_features)
        return outputs.squeeze(1)

    def device(self) -> th.device:
        """Heuristic to determine which device this module is on."""
        first_param = next(self.parameters())
        return first_param.device
