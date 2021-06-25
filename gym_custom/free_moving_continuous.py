import gym
from gym import spaces
import numpy as np
import random
from copy import deepcopy as dc
import pygame
from imitation.data.types import Trajectory
import pickle5 as pickle

class FreeMovingContinuous(gym.Env):

  def __init__(self, speed = 7, framestack = 4, max_steps = np.inf, window_dim = 300):
    super(FreeMovingContinuous, self).__init__()
    self.window_dim = window_dim
    self.framestack = framestack

    self.action_space = spaces.Box(low = -1, high = 1, shape=(2,), dtype=np.float16)
    self.observation_space = spaces.Box(low=-self.window_dim, high=self.window_dim, shape=(self.framestack*2,), dtype=np.float16)
    
    self.speed = speed
    self.max_steps = max_steps
    
    self.cur_steps = 0
    self.agent_pos = None
    self.obs_list = None
    self.reset()

    self.display = None
    self.clock = None

  def step(self, action):
    self.cur_steps += 1
    action = np.array(action, dtype=np.float16)
    action_magnitude = self.getMagnitude(action)

    self.agent_pos += self.speed*action/action_magnitude
    self.agent_pos = np.clip(self.agent_pos, -self.window_dim, self.window_dim)

    self.obs_list.append(dc(self.agent_pos))

    obs = np.array(self.obs_list[-self.framestack:]).reshape(-1)
    done = self.cur_steps>=self.max_steps
    return obs, 0,  done, {}


  def reset(self):
    self.cur_steps = 0
    self.agent_pos = np.array([0,0], dtype=np.float16)
    self.obs_list = []

    for i in range(self.framestack):
      self.obs_list.append(dc(self.agent_pos))

    obs = np.array(self.obs_list[-self.framestack:]).reshape(-1)
    return obs

  def render(self, mode='human', close=False, leave_line = True):
    if not self.display:
      self.display = pygame.display.set_mode((self.window_dim*2, self.window_dim*2))
      self.clock = pygame.time.Clock()

    self.display.fill((255,255,255))

    if leave_line:
      for a_pos in self.obs_list:
        agent_pos_display = (np.round(a_pos) + self.window_dim).astype(int)
        pygame.draw.circle(self.display, (255,0,0), agent_pos_display, 3)
    else:
        agent_pos_display = (np.round(self.agent_pos) + self.window_dim).astype(int)
        pygame.draw.circle(self.display, (255,0,0), agent_pos_display, 3)

    pygame.display.update()
    self.clock.tick(100)

    return None

  def getMagnitude(self, a):
    return (a[0]**2 + a[1]**2)**0.5

  def generateCircleTraj(self, traj_length, radius = 100, direction = [0, -1], noise = 0.5, render = True):
    
    direction = np.array(direction)
    center = radius*direction

    prev_tangent = np.array([1,0])
    action = np.array([0,0], dtype = np.float16)
    action_decay = 0.8

    action_list = []
    obs_list = []
    info_list = []
    traj_dataset = []
    obs = self.reset()
    obs_list.append(obs)

    for i in range(traj_length):
      #Calculating Normal
      normal = self.agent_pos - center
      if self.getMagnitude(normal)>radius:
        normal = -normal
      normal = normal/self.getMagnitude(normal)

      #Calculate tangent direction
      if normal[1]==0:
        y2 = 1
        x2 = -normal[1]/normal[0]
      else:
        x2 = 1
        y2 = -normal[0]/normal[1]
      tangent = np.array([x2,y2])

      if self.getMagnitude(prev_tangent+tangent)<self.getMagnitude(prev_tangent-tangent):
        tangent = -tangent

      prev_tangent = dc(tangent)
      

      #Adding Normal and Tangent with noise
      action = action_decay*action + tangent + normal + np.random.normal(0, noise, size = (2,))
      action = action/self.getMagnitude(action) 
      
      
      obs, _, done, info = self.step(action)
      action_list.append(action)
      obs_list.append(obs)
      info_list.append(info)
      # print(self.agent_pos, action)
      if render:
        self.render()

    
    traj_dataset.append(Trajectory(obs = np.array(obs_list), acts= np.array(action_list), infos = np.array(info_list)))
    self.reset()
    return traj_dataset



class CoverAllTargets(FreeMovingContinuous):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.radius = 100
    self.center = np.array([0,-100])
    
    self.targets = np.array([[70,70]])#np.array([[50,-50], [100,-100], [50,-150], [0,-200], [-50,-150], [-100,-100], [-50,-50], [0,0]])
    self.target_iter = 0

  def step(self, action):
    obs, reward, done, info = super(CoverAllTargets, self).step(action)

    if self.target_iter < len(self.targets) and self.getMagnitude(self.agent_pos-self.targets[self.target_iter])<20:
      reward = 5
      self.target_iter += 1
    
    if self.target_iter >= len(self.targets):
      done = True

    return obs, reward, done, info

  def reset(self):
    obs = super(CoverAllTargets, self).reset()
    self.target_iter = 0
    
    return obs

if __name__=="__main__":
  env = CoverAllTargets()
  traj_dataset = env.generateCircleTraj(1000)  

  # with open('traj_datasets/free_moving_circle.pkl', 'wb') as handle:
  #   pickle.dump(traj_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)