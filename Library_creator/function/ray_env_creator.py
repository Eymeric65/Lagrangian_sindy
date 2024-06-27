import gymnasium as gym

import numpy as np
from .Dynamics_modeling import *

def rk4_step(f, t, y, h,f_ext):
    k1 = h * f(t, y,f_ext)
    k2 = h * f(t + h / 2, y + k1 / 2,f_ext)
    k3 = h * f(t + h / 2, y + k2 / 2,f_ext)
    k4 = h * f(t + h, y + k3,f_ext)
    y_next = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y_next

class MyFunctionEnv(gym.Env):
    def __init__(self,config):
        #uper(MyFunctionEnv, self).__init__()
        # Define action and observation space
        self.action_space = gym.spaces.Box(-2.0, 2.0, shape=(config["coord_numb"],), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(config["coord_numb"]*2,), dtype=np.float32)
        self.state = np.zeros(config["coord_numb"])
        self.target = config["target"]
        self.dyn_function = Dynamics_f_extf(config["dynamics_function_h"])
        self.h=config["h"]
        self.time=0

    def integration(self, input_vector):
        # Replace this with your actual function
        res = rk4_step(self.dyn_function,self.time,self.state,self.h,input_vector)
        self.time = self.time + self.h
        return res

    def reset(self,*, seed=None, options=None ):
        # Reset the state of the environment to an initial state
        self.state = np.zeros(4)
        return self.state, {}

    def step(self, action):
        # Execute one time step within the environment
        self.state = self.integration(action)
        reward = self.compute_reward(self.state)
        done = np.allclose(self.state, self.target, atol=1e-2)  # Done if state is close to target
        truncated = False
        return self.state, reward, done, truncated, {}

    def compute_reward(self, state):
        # Reward is the negative distance to the target
        distance = np.linalg.norm(state - self.target)
        reward = -distance
        return reward

    def render(self, mode='human'):
        pass  # Optional: implement this if you need to render your environment

    def close(self):
        pass