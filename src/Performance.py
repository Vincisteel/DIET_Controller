from typing import Dict, List, Tuple, Any, Callable

from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd

from utils import all_combinations_list
from environment.Environment import Environment
from agent.Agent import Agent


# THE IDEA IS TO FIRST FIND THE BEST CONTROLLER USING ACROSS_TIME
# THEN GIVEN THE FIXED BEST CONFIGURATION, EXAMINE ITS SENSITIVITY TO HYPERPARAMETERS
# IF SENSITIVE, MAYBE START OVER TO FIND POSSIBLY BETTER CONFIGURATION
# IF NOT SENSITIVE, FINAL STEP IS TO EXAMINE BEHAVIOUR IN FIXED POLICY


# ALREADY DEFINED UTILITY FUNCTIONS
def cumulative_reward(data: pd.DataFrame) -> float:
    return np.cumsum(np.array(data["Reward"]))[-1]


# STATISTICS COMPUTATION


def IQR(arr: np.ndarray):
    return np.quantile(arr, 0.75) - np.quantile(arr, 0.25)


def CVaR(arr: np.ndarray, alpha: float = 0.05):
    VaR = np.quantile(arr, alpha)
    return arr[arr < VaR].mean()


# ACROSS_TIME


def compute_dispersion_across_time(
    data: pd.DataFrame, column: str, window: int
) -> float:

    # First detrend to remove any long-running variation that would add noise to our measure
    row = data[column] - data[column].shift(1)
    row[0] = 0.0

    # Compute rolling inter-quartile range and then take the mean
    iqr = row.rolling(window=window).aggregate(lambda x: IQR(np.array(x))).mean()

    return iqr


def compute_risk_across_time(
    data: pd.DataFrame, column: str, alpha: float = 0.05
) -> float:

    # First detrend to remove any long-running variation that would add noise to our measure
    row = data[column] - data[column].shift(1)
    row[0] = 0.0

    return CVaR(np.array(row), alpha=alpha)


def across_time(
    data: pd.DataFrame,
    utility_function: Callable[[pd.DataFrame], float] = cumulative_reward,
    window: int = 1000,
    column: str = "action",
    alpha: float = 0.05,
) -> Tuple[float, float, float]:

    utility = utility_function(data)

    # compute dispersion and risk over a sliding window of time (e.g. window = 1 day)
    # WHY USING A WINDOW ?
    # such that the computed values still makes sense. Indeed, we want to assess how stable and usable
    # the controller can be when used in real life. Thus, we should assess its behaviour over the span
    # of a day instead of 5 months or longer.

    dispersion = compute_dispersion_across_time(data, column, window)
    risk = compute_risk_across_time(data, column, alpha=alpha)

    return (utility, dispersion, risk)


# ACROSS_RUNS


def across_runs(
    agent: Agent,
    agent_arguments: Dict[str, Any],
    parameter: Tuple[str, List[Any]],
    logging_path: str,
    num_episodes: int,
    num_iterations: int,
    utility_function: Callable[[pd.DataFrame], float] = cumulative_reward,
    alpha=0.05,
):

    # given a set of environment parameters and agent parameters
    # given the set of parameters to vary
    # train the agent for at least 2 episodes in each setting

    # 1.risk and dispersion = take the utility of each run and compute IQR and CVaR on it
    # OR
    # 2. run across_time on each run and then process their IQR and CVaR

    # setting up the paramter grid to be used by all_combinations_list
    parameter_name, parameter_list = parameter
    agent_arguments[parameter_name] = parameter_list

    # list to store utility for each run
    utilities_results = []
    # list to store each result path of each run to be logged later
    results_path_list = []

    for curr_agent_arguments in all_combinations_list(agent_arguments):

        # must reset agent before training it again in this case
        curr_agent: Agent = agent.reset().from_dict(curr_agent_arguments)

        results_path, summary_df = curr_agent.train(
            logging_path=logging_path,
            num_episodes=num_episodes,
            num_iterations=num_iterations,
            log=True,
        )
        utilities_results.append(utility_function(summary_df))
        results_path_list.append(results_path)

    utilities_results = np.array(utilities_results)

    iqr = IQR(utilities_results)
    cvar = CVaR(utilities_results, alpha=alpha)

    results_dict = {
        "parameter": parameter,
        "utility_function": utility_function.__name__,
        "utilities_results": utilities_results.tolist(),
        "results_path_list": results_path_list,
        "IQR": iqr,
        "CVaR": cvar,
    }

    return (iqr, cvar, results_dict)


# agent arguemnts most proabably from agent.log_dict or extract them from the json using log_dict.keys()
# same thing for env_arguments
def across_fixed_policy(
    agent: Agent,
    num_testing: int,
    logging_path: str,
    num_episodes: int,
    utility_function: Callable[[pd.DataFrame], float] = cumulative_reward,
    alpha=0.05,
) -> Tuple[float, float]:

    # list to store utility for each run
    utilities_results = []

    for i in range(num_testing):

        _, summary_df = agent.test(
            logging_path=logging_path,
            num_episodes=num_episodes,
            num_iterations=None,
            log=True,
        )

        utilities_results.append(utility_function(summary_df))

    utilities_results = np.array(utilities_results)

    return (IQR(utilities_results), CVaR(utilities_results, alpha=alpha))

    # MAKE SURE THAT THE DATAFRAMES WE ARE GIVEN WERE FILTERED ON OCCUPANCY FIRST

    # should there be a global function that runs the entire pipeline ?
    # DATA MODEL
    # we should expect dataframes of the length of the training and containing the attributes we want

    # Across time
    # Specifiy the attribute we want to assess

    # Across Runs
    # specify the list of parameters : seeds, hyperparameters, .... to check the reproducibility of the run

    # Across rollouts of a fixed policy
    # perform the tests ourselves by just using a policy
    # need to know the environment parameters, get from the json and set parameters
    # same idea for the agent, load everything from the json,  and put him in test mode, i guess
