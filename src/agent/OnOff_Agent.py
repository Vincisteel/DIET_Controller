from calendar import c
from typing import Dict, List, Tuple, Any
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim


import os
import pandas as pd

from Logger import SimpleLogger

from agent.Agent import Agent
from environment.Environment import Environment


class OnOffAgent(Agent):
    def __init__(
        self, env: Environment, is_step:bool = True,
    ):

        self.env = env

    def from_dict(self, dict_arguments: Dict[str, Any]) -> Agent:
        for k, v in dict_arguments.items():
            setattr(self, k, v)

        return self

    def select_action(self, state: np.ndarray) -> np.ndarray:

        d = self.env.observation_to_dict(state)
        occ = d["Occ"][0]
        if self.is_step:
            selected_action = self.env.min_temp if occ == 0.0 else self.env.max_temp
        else:
            selected_action = self.env.min_temp + occ*(self.env.max_temp - self.env.min_temp)

        return selected_action

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take an action and return the response of the env."""
        next_state, reward, done, info = self.env.step(action)

        return next_state, reward, done, info

    def train(
        self,
        logging_path: str,
        num_iterations=None,
        num_episodes=1,
        log=True,
        is_test=False,
    ) -> Tuple[str, pd.DataFrame]:
        """Train the agent."""
        self.is_test = is_test

        ## check num_iterations
        if num_iterations is None:
            num_iterations = self.env.numsteps

        if num_iterations > self.env.numsteps:
            print(
                f"WARNING: Number of iterations chosen ({num_iterations}) is higher than the number of steps of the environment ({self.env.numsteps}) "
            )
            num_iterations = self.env.numsteps

        ## instantiate logger
        logger = SimpleLogger(
            logging_path=logging_path,
            agent_name="OnOff_Agent",
            num_episodes=num_episodes,
            num_iterations=num_iterations,
        )

        # plotting options (make sure the dictionary is in the same order as the columns of the outputted summary_df)
        self.opts = {
            "Tair": {"secondary_y": None, "range": [10, 24], "unit": "(°C)",},
            "Tset": {
                "secondary_y": "moving_average",
                "range": [14, 22],
                "unit": "(°C)",
            },
            "PMV": {"secondary_y": None, "range": [-3, 3], "unit": "(-)",},
            "Heating": {"secondary_y": "cumulative", "range": None, "unit": "(kJ)",},
            "Reward": {"secondary_y": "cumulative", "range": [-5, 5], "unit": "(-)",},
            "Occ": {"secondary_y": None, "range": None, "unit": "(-)",},
        }

        summary_df: pd.DataFrame = pd.DataFrame()

        tair = []
        actions = []
        pmv = []
        qheat = []
        rewards = []
        occ = []

        for episode_num in range(num_episodes):

            state = self.env.reset()
            ## chdir back to logging path otherwise then we recall train() mutliple times, the  os.getcwd() will have moved
            os.chdir(logging_path)

            for i in range(num_iterations):

                action = self.select_action(state)
                next_state, reward, done, info = self.step(action)

                if i % 1000 == 0:
                    print(f"Iteration{i}")

                ## keeping track of the value we've seen
                rewards.append(reward)
                actions.append(action)
                pmv.append(info["pmv"][0])
                d = self.env.observation_to_dict(next_state)
                tair.append(d["Tair"][0])
                heat = d["Qheat"][0]
                qheat.append(heat)
                occ.append(d["Occ"][0])

                state = next_state

            ## slicing lower and upper bound
            lower = episode_num * num_iterations
            upper = (episode_num + 1) * num_iterations

            summary_df = pd.DataFrame(
                {
                    "Tair": tair[lower:upper],
                    "Tset": actions[lower:upper],
                    "PMV": pmv[lower:upper],
                    "Heating": qheat[lower:upper],
                    "Reward": rewards[lower:upper],
                    "Occ": occ[lower:upper],
                }
            )

            summary_df["Reward"] = summary_df["Reward"].apply(lambda x: float(x[0]))

            if log:
                logger.plot_and_logging(
                    summary_df=summary_df,
                    agent=self,
                    episode_num=episode_num,
                    is_summary=False,
                    opts=self.opts,
                )

        # plot a summary that contatenates all episodes together for a complete overview of the training

        summary_df = pd.DataFrame(
            {
                "Tair": tair,
                "Tset": actions,
                "PMV": pmv,
                "Heating": qheat,
                "Reward": rewards,
                "Occ": occ,
            }
        )

        summary_df["Reward"] = summary_df["Reward"].apply(lambda x: float(x[0]))

        if log and num_episodes > 1:
            logger.plot_and_logging(
                summary_df=summary_df,
                agent=self,
                episode_num=num_episodes,
                is_summary=True,
                opts=self.opts,
            )

        # self.env.close()

        results_path = logger.RESULT_PATH

        return (results_path, summary_df)

    def test(
        self, logging_path: str, num_iterations=None, num_episodes=1, log=True
    ) -> Tuple[str, pd.DataFrame]:

        return self.train(
            is_test=True,
            logging_path=logging_path,
            num_iterations=num_iterations,
            num_episodes=num_episodes,
            log=log,
        )

    def seed_agent(self, seed):
        pass

    def save(self, filename, directory):
        pass

    def load(self, filename, directory):
        pass

    def log_dict(self) -> Dict[str, Any]:
        return {}
