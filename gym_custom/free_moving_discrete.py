import gym
from gym import spaces
import numpy as np
import random
from copy import deepcopy as dc
from imitation.data.types import Trajectory
import pickle5 as pickle
import gym_custom.utils as utils


class FreeMovingDiscrete(gym.Env):
  
    action_map = {
        0 : np.array([1, 0]), #right
        1 : np.array([-1, 0]), #left
        2 : np.array([0, 1]), #up
        3 : np.array([0, -1])  #down
    }

    def __init__(self, speed = 5, framestack = 4, max_steps = np.inf, window_dim = 300):
        super(FreeMovingDiscrete, self).__init__()
        self.window_dim = window_dim
        self.framestack = framestack

        self.action_space = spaces.Discrete(4)
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
        action = self.action_map[action]

        self.agent_pos = self.agent_pos + self.speed*action
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
            self.display = utils.Window('Free_Moving_Discrete')
        
        img  = 255*np.ones(shape=(self.window_dim*2, self.window_dim*2, 3), dtype=np.uint8)

        if leave_line:
            for a_pos in self.obs_list:
                img = self.addAgent(a_pos, img)
        else:
            img = self.addAgent(self.agent_pos, img)

        if mode == 'human':
            self.display.show_img(img)

        return img

    def getMagnitude(self, a):
        return (a[0]**2 + a[1]**2)**0.5

    def addAgent(self, a_pos, img):
        agent_pos_display = (np.round(a_pos) + self.window_dim).astype(int)

        for i in range(-1,2):
            for j in range(-1,2):
                img = self.colorImg(agent_pos_display[1] + i, agent_pos_display[0] + j, [255,0,0], img)   

                # if i==0 and j==0:
                #     print("\t",agent_pos_display[1] + i,agent_pos_display[0] + j)

        return img

    def colorImg(self, x, y, col, img):
        if x> 0 and x < img.shape[0] and y > 0 and y < img.shape[1]:
            img[x, y] = np.array(col) 
        return img
    
    def generateCircleTraj(self, traj_length, radius = 100, direction = [0, -1], noise = 0.5, render = True):
    
        direction = np.array(direction)
        center = radius*direction

        prev_tangent = np.array([1,0])
        action_cont = np.array([0,0], dtype = np.float16)
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
            action_cont = action_decay*action_cont + tangent + normal + np.random.normal(0, noise, size = (2,))
            action_cont = action_cont/self.getMagnitude(action_cont) 
            
            minV = 5
            action = 0
            for a, dir in self.action_map.items():
                m =  self.getMagnitude(action_cont-dir)
                if m < minV:
                    minV = m
                    action = a

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



class CoverAllTargetsDiscrete(FreeMovingDiscrete):
    def __init__(self):
        super().__init__(window_dim=50)

        self.targets = np.array([[40,40], [-40,40]])#np.array([[50,-50], [100,-100], [50,-150], [0,-200], [-50,-150], [-100,-100], [-50,-50], [0,0]])
        self.target_iter = 0

        self.vicinity = 20

    def step(self, action):
        obs, reward, done, info = super(CoverAllTargetsDiscrete, self).step(action)
        
        
        if self.target_iter < len(self.targets):
            dist = np.abs(self.agent_pos-self.targets[self.target_iter])
            if (dist[0]<self.vicinity and dist[1]<self.vicinity):
                reward = 5
                self.target_iter += 1
        
        if self.target_iter >= len(self.targets):
            done = True

        return obs, reward, done, info

    def reset(self):
        obs = super(CoverAllTargetsDiscrete, self).reset()
        self.target_iter = 0
        
        return obs

    def render(self, mode = 'human', close=False, leave_line = False):
        img  = super().render(mode='False', close=close, leave_line=leave_line)

        for i in range(len(self.targets)):
            img = self.addTarget(self.targets[i], img)

        if mode == 'human':
            self.display.show_img(img)

        return img  

    def addTarget(self, target, img):
        target_display_pos = target + self.window_dim
        for i in range(-self.vicinity, self.vicinity+1):
            img = self.colorImg(target_display_pos[1] + self.vicinity, target_display_pos[0] + i, [0,0,255], img)
            img = self.colorImg(target_display_pos[1] - self.vicinity, target_display_pos[0] + i, [0,0,255], img)
            img = self.colorImg(target_display_pos[1] + i, target_display_pos[0] + self.vicinity, [0,0,255], img)
            img = self.colorImg(target_display_pos[1] + i, target_display_pos[0] - self.vicinity, [0,0,255], img)
                        
        return img

if __name__=="__main__":
    env = CoverAllTargetsDiscrete()
    traj_dataset = env.generateCircleTraj(300, radius=20, )

    with open('traj_datasets/free_moving_discrete_circle.pkl', 'wb') as handle:
      pickle.dump(traj_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)