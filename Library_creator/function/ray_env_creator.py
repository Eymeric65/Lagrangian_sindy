import gymnasium as gym

import numpy as np
from .Dynamics_modeling import *

import matplotlib.pyplot as plt
from .Render import *

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
        self.max_act = 0.15
        self.action_space = gym.spaces.Box(-self.max_act, self.max_act, shape=(config["coord_numb"],), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(config["coord_numb"]*2,), dtype=np.float32)
        self.coord_numb = config["coord_numb"]
        self.state = np.zeros(config["coord_numb"]*2)
        self.target = config["target"]
        self.dyn_function = config["dynamics_function_h"]
        self.h=config["h"]
        self.time=0
        self.max_time = 6

        #bricolage sans renderclass
        self.render_init = False
        self.fig = None
        self.timeobj = None
        self.lineobj = None
        # -----------------

    def integration(self, input_vector):
        # Replace this with your actual function
        res = rk4_step(self.dyn_function,self.time,self.state,self.h,input_vector)
        self.time = self.time + self.h
        return res

    def reset(self,*, seed=None, options=None ):
        # Reset the state of the environment to an initial state
        #self.state = np.zeros(config["coord_numb"]*2)
        self.state = np.zeros(self.coord_numb*2)
        self.time = 0
        return self.state, {}

    def step(self, action):
        # Execute one time step within the environment
        self.state = self.integration(action)
        done = np.allclose(self.state, self.target, atol=1e-1)  # Done if state is close to target

        reward = self.compute_reward(self.state,action) + done * 10 * self.max_time/self.h

        if self.time > self.max_time:
            truncated = True
        else:
            truncated = False

        #truncated = False

        return self.state, reward, done, truncated, {}

    def compute_reward(self, state,action):
        # Reward is the negative distance to the target

        distance = np.linalg.norm(state[::2]-self.target[::2])


        speed = np.linalg.norm(state[1::2]-self.target[1::2])

        cross_pen = np.sqrt(distance*np.linalg.norm(state[1::2]-self.max_act) )
        
        envy =   np.sum(action*state[1::2]) *distance /(self.max_act*self.coord_numb)

        #distance = np.linalg.norm(state - self.target)
        #reward = -(5*distance + speed/(distance+1)*2 )
        #reward = -(distance+cross_pen) +envy



        reward = - np.sum(action**2) + distance*(speed-100-distance )

        if distance<0.5:
            reward += 500


        return reward

    def render(self, mode='human'):

        if mode =='human':

            if not self.render_init:

                plt.ion()
                self.fig = plt.figure()
                self.render_init=True
                _, self.lineobj,self.timeobj = Single_pendulum_one_state(self.fig,1,self.state,self.time)   

            Single_pendulum_update(self.lineobj,self.timeobj,1,self.state,self.time)
            plt.pause(self.h)   

    def close(self):
        pass