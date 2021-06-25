import copy
import os

import gym
import numpy as np
import pickle5 as pickle
import torch as th
from gym_minigrid import wrappers
from imitation.algorithms import adversarial, bc
from imitation.data import rollout
from imitation.util import logger, util
from stable_baselines3 import PPO
from stable_baselines3.common import policies
import gym_custom

from utils import env_wrappers, env_utils

with open("traj_datasets/free_moving_discrete_circle.pkl", "rb") as f:
    trajectories = pickle.load(f)

transitions = rollout.flatten_trajectories(trajectories)


#There are some goals in this environment, but they can be ignored for imitation task 
venv = util.make_vec_env(
    'CoverAllTargetsDiscrete-v0', 
    n_envs=1
)

base_ppo = PPO(policies.ActorCriticPolicy, venv, verbose=1, batch_size=100, n_steps=200)

logger.configure("logs/CoverAllTargetsDiscrete-v0")


#WGAIL TRAINER
wgail_trainer = adversarial.WGAIL(
    venv,
    expert_data=transitions,
    expert_batch_size=60,
    gen_algo=base_ppo,
    n_disc_updates_per_round=5,
    normalize_reward=False,
    normalize_obs=False,
    disc_opt_cls = th.optim.RMSprop, 
    disc_opt_kwargs = {"lr":0.00005}
)

# GAIL TRAINER
# wgail_trainer = adversarial.GAIL(
#     venv,
#     expert_data=transitions,
#     expert_batch_size=60,
#     gen_algo=base_ppo,
#     n_disc_updates_per_round=2,
#     normalize_reward=False,
#     normalize_obs=False
# )

wgail_trainer.train(60000)

for traj in range(10):
    obs = venv.reset()
    venv.render()
    for i in range(100):
        action, _ = wgail_trainer.gen_algo.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)
        venv.render()
        if done:
            break
    print("done")
