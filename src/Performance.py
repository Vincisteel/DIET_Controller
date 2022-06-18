from typing import Dict, List, Tuple, Any, Callable
from pathlib import Path
from collections import namedtuple
import numpy as np
import pandas as pd
from agent.Agent import Agent
from typing import Dict, List, Tuple, Any
import json
import os
from pathlib import Path
from collections import namedtuple
from itertools import product
import numpy as np
import pandas as pd

Parameter = namedtuple("Parameter", ["name", "value"])


# UTILITY FUNCTIONS
def cumulative_reward(data: pd.DataFrame) -> float:
    """ Given a dataframe containing a reward column, computes cumulative reward"""
    if "Reward" in data.columns:
        return np.cumsum(np.array(data["Reward"]))[-1]
    else:
        return np.cumsum(np.array(data["reward"]))[-1]


def negative_cumulative_heating(data: pd.DataFrame) -> float:
    """ Given a dataframe containing a reward column, computes cumulative reward"""

    if "Heating" in data.columns:
        return -np.cumsum(np.array(data["Heating"]))[-1]
    else:
        return -np.cumsum(np.array(data["heating"]))[-1]


def constant_utility(data: pd.DataFrame) -> float:
    """ Constant utility function, to be used when there is no preference between two sessions"""
    return 1.0


# STATISTICS COMPUTATION


def IQR(arr: np.ndarray) -> float:
    """ Computes the inter-quartile range of an array"""
    return np.quantile(arr, 0.75) - np.quantile(arr, 0.25)


def CVaR(arr: np.ndarray, alpha: float = 0.05) -> float:
    """ Computes the conditional value at risk of an array with risk threshold alpha"""
    VaR = np.quantile(arr, alpha)
    filtered =  arr[arr < VaR]
    return filtered.mean() if len(filtered) > 0 else arr.min()


# ACROSS_TIME

def compute_dispersion_across_time(
    data: pd.DataFrame, column: str, window: int
) -> float:
    # Compute rolling inter-quartile range and then take the mean
    iqr = data[column].rolling(window=window).aggregate(lambda x: IQR(np.array(x))).mean()

    return iqr


def compute_risk_across_time(
    data: pd.DataFrame, column: str, alpha: float = 0.05
) -> float:

    return CVaR(np.array(data[column]), alpha=alpha)


def across_time(
    data: pd.DataFrame,
    window: int = 3 * 6,
    column: str = "Tset",
    alpha: float = 0.05,
) -> Tuple[float, float, float]:

    # compute dispersion and risk over a sliding window of time (e.g. window = 1 day)
    # WHY USING A WINDOW ?
    # such that the computed values still makes sense. Indeed, we want to assess how stable and usable
    # the controller can be when used in real life. Thus, we should assess its behaviour over the span
    # of a day instead of 5 months or longer.

    dispersion = compute_dispersion_across_time(data, column, window)
    risk = compute_risk_across_time(data, column, alpha=alpha)

    return (dispersion, risk)


# measuring how much the "optimal" parameters are sensible to stochasticity in the training progress
def across_runs(
    agent: Agent,
    agent_config_path:str, #absolute path to the logging of the agent such that its configuration can be loaded
    parameter: Tuple[str, List[Any]],
    num_episodes: int,
    num_iterations: int = None,
    column_names:List[str] = ["Tset"], # list of the columns in summary df on which we wish to measure risk and dispersion
    utility_function: Callable[[pd.DataFrame], float] = cumulative_reward,
    window:int = 3*6, # window to compute iqr and cvar across time
    alpha=0.05,
):

    agent = load_trained_agent(agent,agent_config_path).reset()

    parameter_name, parameter_list = parameter

    # list to store utility for each run
    utilities_results = []

    # dictionary for results for each column
    column_names_results_dict = {}

    for name in column_names:
        column_names_results_dict[f"{name}_dispersion"] = []
        column_names_results_dict[f"{name}_risk"] = []


    for parameter_value in parameter_list:

        # must reset agent before training it again in this case
        curr_agent: Agent = agent.from_dict({parameter_name:parameter_value})

        results_path, summary_df = curr_agent.train(
            logging_path="",
            num_episodes=num_episodes,
            num_iterations=num_iterations,
            log=False,
        )
        utilities_results.append(utility_function(summary_df))

        # now computing across_time risk and dispersion for each column
        for name in column_names:
            dispersion, risk = across_time(data=summary_df, window=window,column=name,alpha=alpha)
            column_names_results_dict[f"{name}_dispersion"].append(dispersion)
            column_names_results_dict[f"{name}_risk"].append(risk)

    utilities_results = np.array(utilities_results)

    dispersion = IQR(utilities_results)
    risk = CVaR(utilities_results, alpha=alpha)

    results_dict = {
        "performance_test":"across_runs",
        "agent_config_path": agent_config_path,
        "parameter_name": parameter_name,
        "parameter_list": parameter_list,
        "window": window,
        "utility_function": utility_function.__name__,
        "utilities_results": utilities_results.tolist(),
        "utility_dispersion":dispersion,
        "utility_risk": risk,
        "column_names": column_names,
    }

    for name in column_names:
        results_dict[f"{name}_dispersion"] = column_names_results_dict[f"{name}_dispersion"]
        results_dict[f"{name}_risk"] = column_names_results_dict[f"{name}_risk"]

    return results_dict


def across_fixed_policy(
    agent: Agent,
    agent_config_path:str, #absolute path to the logging of the agent such that its configuration can be loaded
    num_testing: int,
    num_episodes: int,
    num_iterations:int = None,
    column_names:List[str] = ["Tset"], # list of the columns in summary df on which we wish to measure risk and dispersion
    utility_function: Callable[[pd.DataFrame], float] = cumulative_reward,
    window:int = 3*6, # window to compute iqr and cvar across time
    alpha=0.05,
) -> Tuple[float, float]:

    agent = load_trained_agent(agent,agent_config_path)

    # list to store utility for each run
    utilities_results = []
    column_names_results_dict = {}

    for name in column_names:
        column_names_results_dict[f"{name}_dispersion"] = []
        column_names_results_dict[f"{name}_risk"] = []


    for _ in range(num_testing):

        results_path, summary_df = agent.test(
            logging_path="",
            num_episodes=num_episodes,
            num_iterations=num_iterations,
            log=False,
        )
        utilities_results.append(utility_function(summary_df))

        # now computing across_time risk and dispersion for each column
        for name in column_names:
            dispersion, risk = across_time(data=summary_df, window=window,column=name,alpha=alpha)
            column_names_results_dict[f"{name}_dispersion"].append(dispersion)
            column_names_results_dict[f"{name}_risk"].append(risk)

    utilities_results = np.array(utilities_results)

    dispersion = IQR(utilities_results)
    risk = CVaR(utilities_results, alpha=alpha)

    results_dict = {
        "performance_test":"fixed_policy",
        "agent_config_path": agent_config_path,
        "window": window,
        "utility_function": utility_function.__name__,
        "utilities_results": utilities_results.tolist(),
        "utility_dispersion":dispersion,
        "utility_risk": risk,
        "column_names": column_names,
    }

    for name in column_names:
        results_dict[f"{name}_dispersion"] = column_names_results_dict[f"{name}_dispersion"]
        results_dict[f"{name}_risk"] = column_names_results_dict[f"{name}_risk"]

    return results_dict

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


# Example of how to use search_paths:
# searching_directory = r"C:\Users\DIET_Controller"
#
#  conditions={
#      "alpha": ["<",20], # sessions where alpha was less than 20
#      "beta": [">",2], # sessions where beta was bigger than 2
#      "num_iterations": ["=",21744], # sessions where the whole simulation episode was used
#      "is_test": ["=", True] # only testing sessions
#  }
#
# ## This example is specific to SimpleEnvironment
# ## One may also define
#  conditions["pmv"] = {
#          "[-inf,-2]": ["<",0.2], # less than 20% of the time spent in the [-inf,-2] pmv interval
#          "[-0.5,0.0]": [">", 0.5] # more than 50% of the time spent in the [-0.5,0.0] pmv interval
#  }
# Possible intervals are = ['[-inf,-2]', '[-2.0,-1.5]', '[-1.5,-1.0]',
#  '[-1.0,-0.5]', '[-0.5,0.0]', '[0.0,0.5]', '[0.5,1.0]', '[1.0,inf]']
#
# ## This will return the list of absolute paths of log folders satisfying the above conditions.
#  path_list = search_paths(searching_directory,conditions)


def search_paths(
    searching_directory: str,
    conditions: Dict[str, Any],
    utility_function: Callable[[pd.DataFrame], float] = constant_utility,
    top_k: int = -1,
    normalized: bool = False,
) -> List[str]:
    """ Finds all absolute paths in searching_directory of agent sessions that satisfy the specified conditions.
        Outputs top_k (all if -1) absolute paths ranked according to the utility function
    Args:
        searching_directory (str): Absolute path of the relevant directory where the logs of interest may be found
        conditions (Dict[str,Any]): Conditions that the session must satisfy. Further details on how to define them below.
        utility_function (Callable[[pd.DataFrame], float], optional): Utility function to rank sessions.
        Defaults to constant_utility i.e. no ranking.
        top_k (int, optional): Number of outputted paths. Defaults to 0. If -1, every path is outputted.
        normalized (bool, optional): If True, then the utility function output is divided (i.e. "normalized) by the number of episodes 
        of the session. Defaults to False. If False, does nothing.

    Returns:
        List[str]: All the absolute paths of the sessions logs that satisfy the defined conditions.
    """

    ## list of paths satisfiying the conditions
    path_list = []
    utility_list = []

    for path in Path(searching_directory).glob("**/*json"):

        if os.path.getsize(path) > 0 and str(path).__contains__("env_params"):
            with open(path) as f:
                log_dict = json.load(f)

                ## boolean to check whether the given path satisfies the conditions
                failed = False
                for k in conditions:
                    # pmv intervals are a different logic
                    if k != "pmv":
                        a = log_dict[k]
                        comparator, b = conditions[k]
                        if not (comparison(a, b, comparator=comparator)):
                            failed = True
                            break
                    else:
                        # checking all specified intervals
                        for interval in conditions["pmv"]:
                            a = log_dict["pmvs"][interval]
                            comparator, b = conditions["pmv"][interval]
                            if not (comparison(a, b, comparator=comparator)):
                                failed = True
                                break

                ## conditions are satisfied
                if not (failed):

                    df = load_summary_df(path.parent)
                    if not(df.empty):
                        res = utility_function(df)
                        res = res if not (normalized) else res / log_dict["num_episodes"]
                        utility_list.append(res)
                        path_list.append(path)



    path_list = np.array(path_list)
    utility_list = np.array(utility_list)

    if top_k != -1:
        path_list = path_list[np.flip(np.argsort(utility_list))[:top_k]]
    else:
        path_list = path_list[np.flip(np.argsort(utility_list))]

    return [str(path_name.parent) for path_name in path_list]


def comparison(a, b, comparator: str) -> bool:
    """ Simple utility function used by search_paths()"""
    if comparator == "=":
        return a == b
    elif comparator == "<":
        return a < b
    elif comparator == ">":
        return a > b
    else:
        print("Unsupported operation")
        return False


def all_combinations_list(arguments: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Given a dictionary of type Dict[str, List[Any]], outputs a list of all the combinatorial combinations
    (cartesian product) of the elements in the list. This is useful in the case of trying many different 
    combinations of parameter for a reinforcement learning agent.

    Example:
    Given arguments: {"a":[1,2], "b":[3,4]}

    Outputs: [{"a":1, "b":3}, {"a":1, "b":4}, {"a":2, "b":3}, {"a":2, "b":4}]
    

    Args:
        arguments (Dict[str, List[Any]]): Dictionary containing key-value pairs where the key is a string 
        and the value is a list of parameters.

    Returns:
        List[Dict[str,Any]]: Cartesian product of the elements.
    """
    parameter_space = []
    argument_list = []

    for param, values in arguments.items():
        parameters = []
        if not isinstance(values, list) and not isinstance(values, np.ndarray):
            values = [values]

        for value in values:
            # Convert other sequences to tuple to make the parameter accesible to be used as a dictionary key
            parameters.append(
                Parameter(
                    name=param, value=value if np.isscalar(value) else tuple(value)
                )
            )

        parameter_space.append(parameters)

    # unpacking and doing the cartesian product
    for params in product(*parameter_space):
        argument_list.append({param.name: param.value for param in params})

    return argument_list





def search_similar(searching_directory: str, subset_log_dict:Dict[str,Any]) -> bool:

    for path in Path(searching_directory).glob("**/*json"):
        if os.path.getsize(path) > 0 and str(path).__contains__("env_params"):
            with open(path) as f:
                curr_log_dict = json.load(f)
                ## counting the number of key-values equalities
                num_similar = [1 for k in subset_log_dict if (k in curr_log_dict)  and (subset_log_dict[k] == curr_log_dict[k])]
                
                # subset_log_dict is a subset of curr_log_dict
                if len(num_similar) == len(subset_log_dict):
                    return True

    return False

def load_json_params(results_path:str)->Dict[str,Any]:
    log_dict={}
    for path in Path(results_path).glob("**/*json"):
        if os.path.getsize(path) > 0 and str(path).__contains__("env_params"):
            with open(path) as f:
                log_dict=json.load(f)

    return log_dict


def load_summary_df(results_path:str) -> pd.DataFrame:
    df = pd.DataFrame({})
    # one has to be careful with generators because
    # they may be consumed only once, thus we
    # need to recreate them
    ## if no summary csv, possibly there was only one episode
    if len(list(Path(results_path).glob("**/*_summary.csv"))) > 0:
        df = pd.read_csv(
            [
                str(curr)
                for curr in Path(results_path).glob("**/*_summary.csv")
            ][0]
        )
    else:
        ## does the csv exist ?
        if len(list(Path(results_path).glob("**/*_1.csv"))) > 0:
            df = pd.read_csv(
                [
                    str(curr)
                    for curr in Path(results_path).glob("**/*_1.csv")
                ][0]
            )

    return df


def load_trained_agent(agent:Agent, results_path:str) -> Agent:

    agent_log_dict = agent.log_dict()

    results_log_dict = load_json_params(results_path)

    new_params = {k:results_log_dict[k] for k in agent_log_dict if k in results_log_dict}

    agent = agent.reset().from_dict(new_params).reset()


    if len(list(Path(results_path).glob("**/torch_ep_summary*.pth"))) > 0:
        for path in Path(results_path).glob("**/torch_ep_summary*.pth"):
            agent.load(directory=str(path.parent), filename="torch_ep_summary")

            return agent

    elif  len(list(Path(results_path).glob("**/torch_ep_1*.pth"))) > 0:
            for path in Path(results_path).glob("**/torch_ep_1*.pth"):
                agent.load(directory=str(path.parent), filename="torch_ep_1")
                return agent

    else:
        return None
