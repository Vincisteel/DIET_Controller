import gym
import sys
import os
from gym.utils import seeding
from gym.spaces import Discrete, Box
import subprocess
import numpy as np
import math
from typing import Dict, List, Tuple


class TrnsysEnv(gym.Env):

    """
    Custom Environment to interact with TRNSYS and define observation and action space
    """

    def __init__(self,
    observation_dim:int = 9,
    action_dim:int = 100, 
    min_temp:int = 16, 
    max_temp:int = 21, 
    starting_obs =None,
    alpha:float = 0.5,
    beta:float = 1):


        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.beta = beta

        ## discretizing the continuous temperature interval
        ## and defining action and observation spaces

        self.action_space = Discrete(action_dim)
        self.observation_space = Box(low=-np.inf,high=np.inf,shape=(9,))

      

        ## mapping between discrete space and temperature
        self.action_to_temp = np.linspace(min_temp,max_temp,action_dim)


        ## keeping track of current state
        if starting_obs is None:
            starting_obs = np.array([20.0, 50.0, 20.0, 0.1, 5.5, 1.0, 1.0, 0.0001, 0.0001])

        self.default_obs = starting_obs
        self.curr_obs = starting_obs


    def step(self, action: Discrete) -> Tuple[Box,float,bool,dict]:
        """

        Run one timestep of the environment’s dynamics. When end of episode is reached, you are responsible for calling reset() to reset this environment’s state. 
        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:

        action(object): an action provided by the agent. 

        Return:

        observation (object): agent’s observation of the current environment. This will be an element of self.observation_space, representing the HVAC environment dynamics
        reward (float) : amount of reward returned after previous action
        done (bool): whether the episode has ended, in which case further step() calls will return undefined results. 
        info (dict): contains auxiliary diagnostic information (helpful for debugging, learning, and logging). 
        """

        reward = self.compute_reward(self.curr_obs,self.alpha,self.beta)

        next_state = self.simulate_trnsys(self.curr_obs,action)
        
        ## defines whether it's time to reset the environemnt or not
        done = False
        ## debugging dict
        info = {}
        return next_state, reward, done, info



    def reset(self) -> Box:
        """
        Resets the environment to an initial state and returns an initial observation.

        Returns:
            np.array: Element of self.observation_space, representing the HVAC environment dynamics
        """
        self.curr_obs = self.default_obs

        return self.curr_obs


    def simulate_trnsys(self, obs:Box, action:Discrete) -> Box:
        ## TODO

        next_state = self.observation_space.sample()

        return next_state



    def observation_to_dict(self,obs:Box) -> Dict[str, float]:
        """
        Given an np.array of the current observation, returns a dictionary with the key being the string description of each element

        Args:
            obs (np.array): observation of the environment, must be an element of self.observation_space
        """
        
        dict_values = {
            "tair_in":obs[0],
            "rh_in": obs[1],
            "tmrt_in": obs[2],
            "vair_in": obs[3],
            "tout_in":obs[4],
            "clo_in": obs[5],
            "met_in": obs[6],
            "occ_in": obs[7],
            "qheat_in": obs[8],
        }

        return dict_values


    def compute_reward(self,obs: Box, alpha: float, beta:float):
        """
        Given an observation of the environment, computes the reward
        based on energy consumption and thermal comfort 

        Args:
            obs (Box): observation of the environment
            alpha (float): parameter for thermal comfort
            beta (float): parameter for energy consumption

        Returns:
            _type_: _description_
        """
        pmv = self.comfPMV(obs)

        dict_values= self.observation_to_dict(obs)
        qheat_in = dict_values["qheat_in"]
        occ_in = dict_values["occ_in"]

        reward = beta * (1 - (qheat_in/15000)) + alpha * (1 - ((pmv + 0.5) ** 2)) * occ_in

        return reward


    def comfPMV(self, obs:Box):
        
        dict_values= self.observation_to_dict(obs)

        ta = dict_values["tair_in"]
        tr = dict_values["tmrt_in"]
        vel= dict_values["vair_in"]
        rh = dict_values["rh_in"]
        met =dict_values["met_in"]
        clo =dict_values["clo_in"]
        wme=0

        pa = rh * 10 * math.exp(16.6536 - 4030.183 / (ta + 235))

        icl = 0.155 * clo  # thermal insulation of the clothing in M2K/W
        m = met * 58.15  # metabolic rate in W/M2
        w = wme * 58.15  # external work in W/M2
        mw = m - w  # internal heat production in the human body
        if icl <= 0.078:
            fcl = 1 + (1.29 * icl)
        else:
            fcl = 1.05 + (0.645 * icl)

        # heat transfer coefficient by forced convection
        hcf = 12.1 * math.sqrt(vel)
        taa = ta + 273
        tra = tr + 273
        # we have verified that using the equation below or this tcla = taa + (35.5 - ta) / (3.5 * (6.45 * icl + .1))
        # does not affect the PMV value
        tcla = taa + (35.5 - ta) / (3.5 * icl + 0.1)

        p1 = icl * fcl
        p2 = p1 * 3.96
        p3 = p1 * 100
        p4 = p1 * taa
        p5 = (308.7 - 0.028 * mw) + (p2 * math.pow(tra / 100.0, 4))
        xn = tcla / 100
        xf = tcla / 50
        eps = 0.00015

        n = 0
        while abs(xn - xf) > eps:
            xf = (xf + xn) / 2
            hcn = 2.38 * math.pow(abs(100.0 * xf - taa), 0.25)
            if hcf > hcn:
                hc = hcf
            else:
                hc = hcn
            xn = (p5 + p4 * hc - p2 * math.pow(xf, 4)) / (100 + p3 * hc)
            n += 1
            if n > 150:
                print('Max iterations exceeded')
                return 1  # fixme should not return 1 but instead PMV=999 as per ashrae standard

        tcl = 100 * xn - 273

        # heat loss diff. through skin
        hl1 = 3.05 * 0.001 * (5733 - (6.99 * mw) - pa)
        # heat loss by sweating
        if mw > 58.15:
            hl2 = 0.42 * (mw - 58.15)
        else:
            hl2 = 0
        # latent respiration heat loss
        hl3 = 1.7 * 0.00001 * m * (5867 - pa)
        # dry respiration heat loss
        hl4 = 0.0014 * m * (34 - ta)
        # heat loss by radiation
        hl5 = 3.96 * fcl * (math.pow(xn, 4) - math.pow(tra / 100.0, 4))
        # heat loss by convection
        hl6 = fcl * hc * (tcl - ta)

        ts = 0.303 * math.exp(-0.036 * m) + 0.028
        pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
        ppd = 100.0 - 95.0 * math.exp(-0.03353 * pow(pmv, 4.0) - 0.2179 * pow(pmv, 2.0))

        return pmv