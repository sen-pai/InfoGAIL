import logging
from typing import Iterable, Mapping, Optional, Union

import gym
import numpy as np
import torch as th

from stable_baselines3.common import on_policy_algorithm, vec_env

from imitation.data import types
from imitation.rewards import discrim_nets
from imitation.algorithms.adversarial import AdversarialTrainer

from .cnn_discriminator import ActObsCNN


class CNNGAIL(AdversarialTrainer):
    def __init__(
        self,
        venv: vec_env.VecEnv,
        expert_data: Union[Iterable[Mapping], types.Transitions],
        expert_batch_size: int,
        gen_algo: on_policy_algorithm.OnPolicyAlgorithm,
        discrim=None,
        *,
        discrim_kwargs: Optional[Mapping] = None,
        **kwargs,
    ):
        """Generative Adversarial Imitation Learning that accepts Image Obs

        Most parameters are described in and passed to `AdversarialTrainer.__init__`.
        Additional parameters that `CNNGAIL` adds on top of its superclass initializer are
        as follows:

        Args:
            discrim_kwargs: Optional keyword arguments to use while constructing the
                DiscrimNetGAIL.

        """
        discrim_kwargs = discrim_kwargs or {}

        if discrim == None:
            discrim = discrim_nets.DiscrimNetGAIL(
                venv.observation_space,
                venv.action_space,
                discrim_net=ActObsCNN,
                **discrim_kwargs,
            )

        logging.info("using CNN GAIL")
        super().__init__(
            venv, gen_algo, discrim, expert_data, expert_batch_size, **kwargs
        )

