# import os
import numpy as np
import time
import argparse

import matplotlib.pyplot as plt

import gym
import gym_minigrid
from gym_minigrid import wrappers

from stable_baselines3 import PPO

from imitation.data.types import Trajectory

import pickle5 as pickle

from utils.env_utils import minigrid_get_env, minigrid_render

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    "-e",
    help="minigrid gym environment to train on",
    default="MiniGrid-LavaCrossingS9N1-v0",
)
parser.add_argument("--run", "-r", help="Run name", default="testing")

parser.add_argument("--save-name", "-s", help="Save name", default="saved_testing")

parser.add_argument(
    "--seed", type=int, help="random seed to generate the environment with", default=1
)
parser.add_argument(
    "--max-timesteps", "-t", type=int, help="cut traj at max timestep", default=50
)

parser.add_argument(
    "--ntraj", type=int, help="number of trajectories to collect", default=10
)

parser.add_argument(
    "--flat",
    "-f",
    default=False,
    help="Partially Observable FlatObs or Fully Observable Image ",
    action="store_true",
)

parser.add_argument(
    "--render", default=False, help="Render", action="store_true",
)

parser.add_argument(
    "--best", default=True, help="Use best model", action="store_false",
)


args = parser.parse_args()


env = minigrid_get_env(args.env, 1, args.flat)

best_model_path = "./logs/" + args.env + "/ppo/" + args.run + "/best_model/best_model.zip"
pkl_save_path = "./traj_datasets/" + args.save_name + ".pkl"


model = PPO.load(best_model_path)

traj_dataset = []
for traj in range(args.ntraj):
    obs_list = []
    action_list = []
    obs = env.reset()
    obs_list.append(obs[0])
    if args.render:
        minigrid_render(obs)

    for i in range(args.max_timesteps):
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = env.step(action)
        action_list.append(action[0])
        obs_list.append(obs[0])

        if args.render:
            minigrid_render(obs)
        if done:
            break
    traj_dataset.append(
        Trajectory(
            obs=np.array(obs_list),
            acts=np.array(action_list),
            infos=np.array([{} for i in action_list]),
        )
    )


with open(pkl_save_path, "wb") as handle:
    pickle.dump(traj_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


print(f"{len(traj_dataset)} trajectories saved at {pkl_save_path}")