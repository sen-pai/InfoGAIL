# import os
import numpy as np
import time
import argparse

import matplotlib.pyplot as plt

import gym
import gym_minigrid
from gym_minigrid import wrappers
from gym_minigrid.window import Window

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

parser.add_argument("--save-name", "-s", help="Run name", default="saved_testing")


parser.add_argument(
    "--flat",
    "-f",
    default=False,
    help="Partially Observable FlatObs or Fully Observable Image ",
    action="store_true",
)

parser.add_argument(
    "--tile_size", type=int, help="size at which to render tiles", default=32
)
parser.add_argument(
    "--agent_view",
    default=False,
    help="draw the agent sees (partially observable view)",
    action="store_true",
)

args = parser.parse_args()

print("After you are done, click ESC to write the data, dont just close the window.")


env_kwargs = {}
if "FourRooms" in args.env:
    env_kwargs = {"agent_pos": (3, 3), "goal_pos": (15, 15)}

env = minigrid_get_env(args.env, 1, args.flat, env_kwargs)

pkl_save_path = "./traj_datasets/" + args.save_name + ".pkl"


traj_dataset = []
obs_list = []
action_list = []


def redraw(img):
    if not args.agent_view:
        img = env.render("rgb_array")

    window.show_img(img)


def reset():
    global obs_list, action_list, info_list
    obs_list = []
    action_list = []

    obs = env.reset()
    obs_list.append(obs[0])

    redraw(obs)


def step(action_int):
    done = False
    if action_int != -1:

        global obs_list, action_list, traj_dataset
        obs, reward, done, _ = env.step([action_int])
        action_list.append(action_int)

        obs_list.append(obs[0])
        print("reward=%.2f" % (reward))

    if done or action_int == -1:
        print("done!")
        print(len(action_list), len(obs_list))
        traj_dataset.append(
            Trajectory(
                obs=np.array(obs_list),
                acts=np.array(action_list),
                infos=np.array([{} for i in action_list]),
            )
        )
        reset()
    else:
        redraw(obs)


def key_handler(event):
    print("pressed", event.key)

    if event.key == "escape":
        window.close()
        return

    if event.key == "backspace":
        step(-1)
        return

    if event.key == "left":
        # step(env.actions.left, 0)
        step(0)

        return
    if event.key == "right":
        # step(env.actions.right, 1)
        step(1)

        return
    if event.key == "up":
        # step(env.actions.forward, 2)
        step(2)
        return

    # Spacebar
    if event.key == " ":
        # step(env.actions.toggle, 5)
        step(5)
        return
    if event.key == "pageup":
        # step(env.actions.pickup, 3)
        step(3)
        return
    if event.key == "pagedown":
        # step(env.actions.drop, 4)
        step(4)
        return

    if event.key == "enter":
        # step(env.actions.done, 6)
        step(6)
        return


window = Window("gym_minigrid - " + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)


with open(pkl_save_path, "wb") as handle:
    pickle.dump(traj_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


print(f"{len(traj_dataset)} trajectories saved at {pkl_save_path}")
