from gym.utils import seeding
from gym.spaces import Discrete, Box
import numpy as np
from typing import Any, Dict, List, Tuple

from environment.SimpleEnvironment import SimpleEnvironment


class ContinuousSimpleEnvironment(SimpleEnvironment):
    def __init__(
        self,
        param_list: List[str] = [
            "Tair",
            "RH",
            "Tmrt",
            "Tout",
            "Qheat",
            "Occ",
        ],  # what we get from the model at each step
        min_temp: int = 16,  # minimum temperature for action
        max_temp: int = 21,
        alpha: float = 1,  # thermal comfort
        beta: float = 1,  # energy consumption
        modelname: str = "CELLS_v1.fmu",
        simulation_path: str = r"C:\Users\Harold\Desktop\ENAC-Semester-Project\DIET_Controller\EnergyPlus_simulations\simple_simulation",
        days: int = 151,
        hours: int = 24,
        minutes: int = 60,
        seconds: int = 60,
        ep_timestep: int = 6,
    ):
        super().__init__(
            param_list,
            min_temp,
            max_temp,
            alpha,
            beta,
            modelname,
            simulation_path,
            days,
            hours,
            minutes,
            seconds,
            ep_timestep,
        )

        self._action_space = Box(low=self.min_temp, high=self._max_temp, shape=(1,))

    @property
    def action_space(self):
        return self._action_space

    @property
    def min_temp(self):
        return self._min_temp

    @min_temp.setter
    def min_temp(self, temp):
        self._min_temp = temp
        self._action_space = Box(low=self.min_temp, high=self._max_temp, shape=(1,))

    @property
    def max_temp(self):
        return self._max_temp

    @max_temp.setter
    def max_temp(self, temp):
        self._max_temp = temp
        self._action_space = Box(low=self.min_temp, high=self._max_temp, shape=(1,))

    def reset(self, seed=42) -> np.ndarray:
        self.action_space.seed(seed)
        return super().reset(seed)

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        return super().step(action=action)

    def log_dict(self):
        log_dict = super().log_dict()

        return log_dict

