from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Any, TypeVar
import numpy as np
from gym.spaces import Space


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class Environment(metaclass=ABCMeta):
    @property
    @abstractmethod
    def action_space(self) -> Space:
        ...

    @property
    @abstractmethod
    def observation_space(self) -> Space:
        ...

    @property
    @abstractmethod
    def action_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def observation_dim(self) -> int:
        ...

    @abstractmethod
    def reset(self) -> np.ndarray:
        pass

    @abstractmethod
    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        pass

    ##@abstractmethod
    ##def from_dict(self, dict_arguments: Dict[str, Any]):
    ##    pass

    @abstractmethod
    def log_dict(self) -> Dict[str, Any]:
        pass

