from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Any
import pandas as pd
import gym


class Agent(metaclass=ABCMeta):
    @abstractmethod
    def from_dict(self, dict_arguments: Dict[str, Any]) -> Agent:
        pass

    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def train(
        self, logging_path: str, num_episodes: int, num_iterations: int, log: bool
    ) -> Tuple[str, pd.DataFrame]:
        pass

    @abstractmethod
    def test(
        self, logging_path: str, num_episodes: int, num_iterations: int, log: bool
    ) -> Tuple[str, pd.DataFrame]:
        pass

    @abstractmethod
    def log_dict(self) -> Dict[str, Any]:
        pass

    def __getattribute__(self, attr):
        return object.__getattribute__(self, attr)

    def __setattr__(self, attr, value):
        object.__setattr__(self, attr, value)

