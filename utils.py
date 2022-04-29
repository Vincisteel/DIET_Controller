import glob
from typing import Dict, List, Tuple, Any
import json
import os
from pathlib import Path
from collections import namedtuple
from itertools import product
import numpy as np

Parameter = namedtuple('Parameter', ['name', 'value'])




# Example of how to use the function: 
# 
# searching_directory = r"C:\\Users\\Harold\\Desktop\\ENAC-Semester-Project\\DIET_Controller"
# 
# conditions={
#     "beta": ["<",20],
#     "alpha": ["<",20]
#     "num_iterations": ["=",21744] # to only have have trainings where the full simulation was used
# }
# 
# conditions["pmv"] = {
#         "[-inf,-2]": [">",0.2]
# }
# 
# path_list = search_paths(searching_directory,conditions)


def search_paths(searching_directory, conditions):

    ## list of paths satisfiying the conditions
    path_list=[]
    reward_list=[]
    cum_heat_list=[]

##intervals are = ['[-inf,-2]', '[-2.0,-1.5]', '[-1.5,-1.0]', '[-1.0,-0.5]', '[-0.5,0.0]', '[0.0,0.5]', '[0.5,1.0]', '[1.0,inf]']
    for path in Path(searching_directory).glob("**/*json"):

        if os.path.getsize(path) > 0 and str(path).__contains__("env_params"):
            with open(path) as f:
                log_dict = json.load(f)

                failed = False
                for k in conditions:
                    if k != "pmv":
                        a = log_dict[k]
                        comparator, b = conditions[k]
                        if not comparison(a,b,comparator=comparator):
                            failed = True
                            break
                    else:
                        for k in conditions["pmv"]:
                            a = log_dict["pmvs"][k]
                        comparator, b = conditions["pmv"][k]
                        if not comparison(a,b,comparator=comparator):
                            failed = True
                            break

                if not failed:
                    path_list.append(path)
                    reward_list.append(log_dict["final_reward"])
                    cum_heat_list.append(log_dict["final_cumulative_heating"])

    
    reward_list=np.array(reward_list)
    cum_heat_list= np.array(cum_heat_list)

    best_reward_path = path_list[np.argmax(reward_list)]
    best_heat_path = path_list[np.argmax(cum_heat_list)]

    return path_list, best_reward_path, best_heat_path



    
def comparison(a,b, comparator:str):

    if comparator == "=":
        return a == b
    elif comparator == "<":
        return a < b
    elif comparator == ">":
        return a > b
    else: 
        print("Unsupported operation")
        return False



def all_combinations_list(arguments:Dict[str, List[Any]]):

    parameter_space = []
    argument_list = []

    for param, values in arguments.items():
        parameters = []
        if not isinstance(values, list) and not isinstance(values, np.ndarray):
            values = [values]

        for value in values:
               # Convert other sequences to tuple to make the parameter accesible to be used as a dictionary key
            parameters.append(Parameter(name=param, value=value if np.isscalar(value) else tuple(value)))

        parameter_space.append(parameters)

    for params in product(*parameter_space):
        argument_list.append({param.name: param.value for param in params})

    return argument_list