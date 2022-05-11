from ast import Call
import glob
from typing import Dict, List, Tuple, Any, Callable
import json
import os
from pathlib import Path
from collections import namedtuple
from itertools import product
import numpy as np
import pandas as pd
import gym
import sys

from utils import all_combinations_list
from DQN_Agent import *
from Logger import *



### THE IDEA IS TO FIRST FIND THE BEST CONTROLLER USING ACROSS_TIME
## THEN GIVEN THE FIXED BEST CONFIGURATION, EXAMINE ITS SENSITIVITY TO HYPERPARAMETERS
## IF SENSITIVE, MAYBE START OVER TO FIND POSSIBLY BETTER CONFIGURATION
## IF NOT SENSITIVE, FINAL STEP IS TO EXAMINE BEHAVIOUR IN FIXED POLICY

def assess_performance(path:str, column = "reward", window = 1000):

    res = {}

    for path in Path(path).glob("**/*.csv"):
        df = pd.read_csv(path)
        # only assessing performance when occupancy isn't zero
        df = df[df.occ != 0.0]

        iqr = df[column].rolling(window=window).aggregate(lambda x: x.quantile(0.75) - x.quantile(0.25)).mean()

        res[path] = iqr

    return res




## ALREADY DEFINED UTILITY FUNCTIONS
def cumulative_reward(data:pd.DataFrame) -> float:
    return np.cumsum(np.array(data["reward"]))[-1]


## STATISTICS COMPUTATION

def IQR(arr:np.ndarray):
    return arr.quantile(0.75) - arr.quantile(0.25)

def CVaR(arr:np.ndarray, alpha:float = 0.05):
    VaR = arr.quantile(alpha)
    return arr[arr < VaR].mean()


## ACROSS_TIME

def compute_dispersion_across_time(data:pd.DataFrame, column:str, window:int) -> float:

    ## First detrend to remove any long-running variation that would add noise to our measure
    row = (data[column] - data[column].shift(1))
    row[0] = 0.0

    ## Compute rolling inter-quartile range and then take the mean
    iqr = row.rolling(window=window).aggregate(lambda x: IQR(np.array(x))).mean()

    return iqr


def compute_risk_across_time(data:pd.DataFrame, column:str, window:int, alpha:float = 0.05) -> float:

    ## First detrend to remove any long-running variation that would add noise to our measure
    row = (data[column] - data[column].shift(1))
    row[0] = 0.0

    return CVaR(np.array(row), alpha=alpha)



def across_time(data:pd.DataFrame, utility_function: Callable[[pd.DataFrame],float] = cumulative_reward,  window:int = 1000, column:str = "action", alpha=0.05) -> Tuple[float,float,float]:

    utility = utility_function(data)

    ## compute dispersion and risk over a sliding window of time (e.g. window = 1 day)
    ## WHY USING A WINDOW ?
    ## such that the computed values still makes sense. Indeed, we want to assess how stable and usable
    ## the controller can be when used in real life. Thus, we should assess its behaviour over the span
    ## of a day instead of 5 months or longer.

    dispersion = compute_dispersion_across_time(data, column, window)
    risk = compute_risk_across_time(data,column,window)

    return (utility,dispersion,risk)


## ACROSS_RUNS


def across_runs(env_name:str, env_arguments:Dict[str,Any],  agent_arguments:Dict[str,Any], parameter:Tuple[str,List[Any]], logging_path:str, num_episodes:int, utility_function: Callable[[pd.DataFrame],float] = cumulative_reward, alpha = 0.05):

    ## given a set of environment parameters and agent parameters
    ## given the set of parameters to vary 
    ## train the agent for at least 2 episodes in each setting

    # 1.risk and dispersion = take the utility of each run and compute IQR and CVaR on it
    # OR 
    # 2. run across_time on each run and then process their IQR and CVaR

    ## setting up the paramter grid to be used by all_combinations_list
    parameter_name, parameter_list = parameter
    agent_arguments[parameter_name] = parameter_list

    # list to store utility for each run
    utilities_results = []

    for curr_agent_arguments in all_combinations_list(agent_arguments):

        ## for example env_name = 'EnergyPlusEnv-v0'
        env = gym.make(env_name)
        env.set_arguments(env_arguments)

        agent = DQNAgent(env, dict_arguments=curr_agent_arguments)

        results_path, summary_df = agent.train(logging_path= logging_path, num_episodes=num_episodes,log=True)
        utilities_results.append(utility_function(summary_df))

    utilities_results = np.array(utilities_results)

    return (IQR(utilities_results), CVaR(utilities_results,alpha=alpha))



### agent arguemnts most proabably from agent.log_dict or extract them from the json using log_dict.keys()
### same thing for env_arguments
def across_fixed_policy(controller_path:str, env_name:str, num_testing:int, utility_function: Callable[[pd.DataFrame],float] = cumulative_reward, alpha = 0.05 ) -> Tuple[float,float]:


    # TODO get arguments from json and add parameter_list to csv
    env_arguments = {}
    agent_arguments= {}

    filename = "torch_ep_summary"
    directory= controller_path + r"\model_weights"

    # list to store utility for each run
    utilities_results = []

    for i in range(num_testing):
        env = gym.make(env_name)
        env.set_arguments(env_arguments)

        agent = DQNAgent(env,dict_arguments=agent_arguments)
        agent.load(filename=filename,directory=directory)
        agent.test(logging_path=)

        utilities_results.append(utility_function(summary_df))

    utilities_results = np.array(utilities_results)

    return (IQR(utilities_results), CVaR(utilities_results,alpha=alpha))






    









    ## MAKE SURE THAT THE DATAFRAMES WE ARE GIVEN WERE FILTERED ON OCCUPANCY FIRST 


    ## should there be a global function that runs the entire pipeline ?

    ## DATA MODEL

    ## we should expect dataframes of the length of the training and containing the attributes we want

    ## Across time
    ## Specifiy the attribute we want to assess

    ## Across Runs 
    ## specify the list of parameters : seeds, hyperparameters, .... to check the reproducibility of the run    


    ## Across rollouts of a fixed policy
    ## perform the tests ourselves by just using a policy
    ## need to know the environment parameters, get from the json and set parameters
    ## same idea for the agent, load everything from the json,  and put him in test mode, i guess

