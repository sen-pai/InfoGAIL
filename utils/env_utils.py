import numpy as np
import matplotlib.pyplot as plt

import gym
import gym_minigrid
from gym_minigrid import wrappers

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage

import torch
import random

from utils import env_wrappers


def minigrid_get_env(
    env, n_envs, flat=False, partial=False, encoder=None, env_kwargs={}
):

    if (not partial) and encoder:
        img_wrappers = lambda env: env_wrappers.EncoderWrapper(
            wrappers.ImgObsWrapper(wrappers.RGBImgObsWrapper(env)), encoder
        )
    elif not partial:
        img_wrappers = lambda env: wrappers.ImgObsWrapper(
            wrappers.RGBImgObsWrapper(env)
        )
    else:
        img_wrappers = lambda env: wrappers.ImgObsWrapper(
            wrappers.RGBImgPartialObsWrapper(env)
        )
    flat_wrapper = lambda env: env_wrappers.FlatObsOnlyWrapper(env)

    vec_env = make_vec_env(
        env_id=env,
        n_envs=n_envs,
        wrapper_class=flat_wrapper if flat else img_wrappers,
        env_kwargs=env_kwargs,
    )

    if flat or encoder:
        return vec_env
    return VecTransposeImage(vec_env)


def minigrid_render(obs):
    if obs.shape == 4:
        plt.imshow(np.moveaxis(obs[0], 0, -1))
    else:
        plt.imshow(np.moveaxis(obs, 0, -1))
    plt.show()
    plt.close()


def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
