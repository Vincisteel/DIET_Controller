import gym
import sys
import os
from gym import error, spaces
from gym.utils import seeding
import subprocess
import numpy as np


def import_path(fullpath):
    """ 
    Import a file with full path specification. Allows one to
    import from anywhere, something __import__ does not do. 
    """
    path, filename = os.path.split(fullpath)
    filename, ext = os.path.splitext(filename)
    sys.path.append(path)
    module = __import__(filename)
    del sys.path[-1]
    return module

utils = import_path("../../../utils.py")

class TrnsysEnv(gym.Env):

    """
    Custom Environment to interact with TRNSYS and define observation and action space
    """

    def __init__(self,
    observation_dim:int = 9,
    action_dim:int = 100, 
    min_temp:int = 16, 
    max_temp:int = 21, 
    starting_state =None,
    alpha:int = 0.5,
    beta:int = 1):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.beta = beta

        ## discretizing the continuous temperature interval
        self.action_space = np.linspace(min_temp,max_temp,action_dim)
        if starting_state is None:
            starting_state = utils.State([20.0, 50.0, 20.0, 0.1, 5.5, 1.0, 1.0, 0.0001, 0.0001])

        self.default_state = starting_state
        self.curr_state = starting_state

    def simulate_trnsys(self, state,action:int):
        ## TODO

        next_state = utils.State([i for i in range(10)])

        return next_state

    def compute_reward(self,state, alpha:int, beta:int):
        pmv = utils.comfPMV(state)
        qheat_in = state.dict_values["qheat_in"]
        occ_in = state.dict_values["occ_in"]
        reward = beta * (1 - (qheat_in/15000)) + alpha * (1 - ((pmv + 0.5) ** 2)) * occ_in

        return reward


    def step(self, action:int):

        reward = self.compute_reward(self.curr_state,self.alpha,self.beta)

        next_state = self.simulate_trnsys(self.curr_state,action)
        
        ## defines whether it's time to reset the environemnt or not
        done = False
        ## debugging dict
        info = {}
        return next_state, reward, done, info

    def reset(self):
        self.curr_state = self.default_state