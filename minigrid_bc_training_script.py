from utils.env_utils import minigrid_render, minigrid_get_env
import os, time
import numpy as np

import argparse

import matplotlib.pyplot as plt


import pickle5 as pickle
from imitation.data import rollout
from imitation.util import logger, util
from imitation.algorithms import bc

import gym
import gym_minigrid


from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy


from imitation.algorithms import bc


parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    "-e",
    help="minigrid gym environment to train on",
    default="MiniGrid-LavaCrossingS9N1-v0",
)
parser.add_argument("--run", "-r", help="Run name", default="testing")

parser.add_argument(
    "--save-name", "-s", help="BC weights save name", default="saved_testing"
)

parser.add_argument("--traj-name", "-t", help="Run name", default="saved_testing")


parser.add_argument(
    "--seed", type=int, help="random seed to generate the environment with", default=1
)

parser.add_argument(
    "--nenvs", type=int, help="number of parallel environments to train on", default=1
)

parser.add_argument(
    "--nepochs", type=int, help="number of epochs to train till", default=50
)


parser.add_argument(
    "--flat",
    "-f",
    default=False,
    help="Partially Observable FlatObs or Fully Observable Image ",
    action="store_true",
)

parser.add_argument(
    "--show", default=False, help="See a sample image obs", action="store_true",
)

parser.add_argument(
    "--vis-trained",
    default=False,
    help="Render 10 traj of trained BC",
    action="store_true",
)


args = parser.parse_args()


train_env = minigrid_get_env(args.env, args.nenvs, args.flat)

if args.show and not args.flat:
    plt.imshow(np.moveaxis(train_env.reset()[0], 0, -1))
    plt.show()

save_path = "./logs/" + args.env + "/bc/" + args.run + "/"
os.makedirs(save_path, exist_ok=True)
traj_dataset_path = "./traj_datasets/" + args.traj_name + ".pkl"

print(f"Expert Dataset: {args.traj_name}")

policy_type = ActorCriticPolicy if args.flat else ActorCriticCnnPolicy

with open(traj_dataset_path, "rb") as f:
    trajectories = pickle.load(f)

transitions = rollout.flatten_trajectories(trajectories)

logger.configure(args.save_name)
bc_trainer = bc.BC(
    train_env.observation_space,
    train_env.action_space,
    is_image=False,
    expert_data=transitions,
    loss_type="original",
    policy_class=policy_type,
)
bc_trainer.train(n_epochs=args.nepochs)
os.chdir(save_path)
bc_trainer.save_policy(args.save_name + ".pt")

if args.vis_trained:
    for traj in range(10):
        obs = train_env.reset()
        train_env.render()
        for i in range(40):
            action, _ = bc_trainer.policy.predict(obs, deterministic=True)

            obs, reward, done, info = train_env.step(action)
            train_env.render()
            if done:
                break

print(f"Weights Saved at {save_path}")
