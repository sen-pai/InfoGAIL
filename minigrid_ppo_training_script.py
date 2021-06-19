# import os

import argparse


import gym
import gym_minigrid

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)

from utils.env_utils import minigrid_get_env, minigrid_render

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    "-e",
    help="minigrid gym environment to train on",
    default="MiniGrid-LavaCrossingS9N1-v0",
)
parser.add_argument("--run", "-r", help="Run name", default="testing")
parser.add_argument(
    "--seed", type=int, help="random seed to generate the environment with", default=1
)
parser.add_argument(
    "--timesteps", "-t", type=int, help="total timesteps to learn", default=1e5
)

parser.add_argument(
    "--nenvs", type=int, help="number of parallel environments to train on", default=1
)


parser.add_argument(
    "--flat",
    "-f",
    default=False,
    help="Partially Observable FlatObs or Fully Observable Image ",
    action="store_true",
)

parser.add_argument(
    "--partial",
    "-p",
    default=False,
    help="Partially Observable Img or Fully Observable Image ",
    action="store_true",
)


parser.add_argument(
    "--show", default=False, help="See a sample image obs", action="store_true",
)

parser.add_argument("--load-env", "-le", help="Env that was before in curr", default="NA")


parser.add_argument("--load", "-l", help="Load weights from another trained PPO", default="NA")

args = parser.parse_args()

train_env = minigrid_get_env(args.env, args.nenvs, args.flat, partial= args.partial)
eval_env = minigrid_get_env(args.env, 1, args.flat, partial= args.partial)

if args.show and not args.flat:
    minigrid_render(train_env.reset())

save_path = "./logs/" + args.env + "/ppo/" + args.run

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=save_path)
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=0.98, verbose=1)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=save_path + "/best_model",
    log_path=save_path + "/eval_results",
    eval_freq=1000,
    callback_on_new_best=callback_on_best,
)
callback = CallbackList([checkpoint_callback, eval_callback])



if args.load != "NA":
    best_model_path = "./logs/" + args.load_env + "/ppo/" + args.load + "/best_model/best_model"
    model = PPO.load(best_model_path, env = train_env)
    print("loaded")
else:
    policy_type = "MlpPolicy" if args.flat else "CnnPolicy"
    model = PPO(policy=policy_type, env=train_env, verbose=1, seed=args.seed)

model.learn(total_timesteps=int(args.timesteps), callback=callback)
