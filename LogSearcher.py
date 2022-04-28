import glob
import json
import os
from pathlib import Path

## possible intervals for the pmv
##'intervals': {0: '[-inf,-2]',
##  1: '[-2.0,-1.5]',
##  2: '[-1.5,-1.0]',
##  3: '[-1.0,-0.5]',
##  4: '[-0.5,0.0]',
##  5: '[0.0,0.5]',
##  6: '[0.5,1.0]',
##  7: '[1.0,inf]'}


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

##interval_mapping = {'[-inf,-2]': 0,
## '[-2.0,-1.5]': 1,
## '[-1.5,-1.0]': 2,
## '[-1.0,-0.5]': 3,
## '[-0.5,0.0]': 4,
## '[0.0,0.5]': 5,
## '[0.5,1.0]': 6,
## '[1.0,inf]': 7}

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

    return path_list



    
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