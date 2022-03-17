import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from pyfmi import load_fmu
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import os
import pandas as pd
import gym
import sys
import os
from gym.utils import seeding
from gym.spaces import Discrete, Box
import subprocess
import numpy as np
import math
from typing import Dict, List, Tuple




# We set the parameters





## DDPG parameters

save_models = True  # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.1  # Exploration noise - STD value of exploration Gaussian noise
batch_size = 128  # Size of the batch
discount = 0.99  # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.05  # Target network update rate
policy_noise = 0.2  # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5  # Maximum value of the Gaussian noise added to the actions (policy)
alpha = 3  # Adjusting co-efficient for the comfort reward
beta = 1  # Adjusting co-efficient for the energy reward


## Environment parameters

modelname = 'CELLS_v1'
days = 151  # Number of days the simulation is run for
hours = 24  # Number of hours each day the simulation is run for
minutes = 60
seconds = 60
ep_timestep = 6  # Number of timesteps per hour
num_random_episodes = 1
num_total_episodes = 3
numsteps = days * hours * ep_timestep
timestop = days * hours * minutes * seconds
secondstep = timestop / numsteps
comfort_lim = 0
min_temp=12
max_temp=30


## initilaize / reset 

model = load_fmu(modelname + '.fmu')
opts = model.simulate_options()  # Get the default options
opts['ncp'] = numsteps  # Specifies the number of timesteps
opts['initialize'] = False
simtime = 0
model.initialize(simtime, timestop)
index = 0
t = np.linspace(0.0, numsteps-1, numsteps)
inputcheck_heating = np.zeros((numsteps, 1))
tair = np.zeros((numsteps, 1))
rh = np.zeros((numsteps, 1))
tmrt = np.zeros((numsteps, 1))
tout = np.zeros((numsteps, 1))
occ = np.zeros((numsteps, 1))
qheat = np.zeros((numsteps, 1))
pmv = np.zeros((numsteps, 1))
reward = np.zeros((numsteps, 1))
action = np.zeros((numsteps, 1))
state = np.zeros((numsteps, 6))



## do a step

model.set('Thsetpoint_diet', action[index])
res = model.do_step(current_t=simtime, step_size=secondstep, new_step=True)
inputcheck_heating[index] = model.get('Thsetpoint_diet')
tair[index], rh[index], tmrt[index], tout[index], qheat[index], occ[index], inputcheck_heating[index] = model.get(['Tair', 'RH', 'Tmrt', 'Tout', 'Qheat', 'Occ', 'Thsetpoint_diet'])
state[index][0], state[index][1], state[index][2], state[index][3], state[index][4], state[index][5] = tair[index], rh[index], tmrt[index], tout[index], occ[index], qheat[index]
pmv[index] = comfPMV(tair[index], tmrt[index], rh[index])
reward[index] = beta * (1 - (qheat[index]/(800*1000))) + alpha * (1 - abs(pmv[index] + 0.5)) * occ[index]   # * int(bool(occ[index])
if index == 0:
    obs = np.array([tair[index], rh[index], tmrt[index], tout[index], occ[index], qheat[index]]).flatten(
else:
    new_obs = np.array([tair[index], rh[index], tmrt[index], tout[index], occ[index], qheat[index]]).flatten()
    # We store the new transition into the Experience Replay memory (ReplayBuffer)
    replay_buffer.add((obs, new_obs, action[index], reward[index]))
    obs = new_ob
simtime += secondstep
index += 1



def simulate_energyplus(self, obs: np.ndarray, action:Discrete) -> np.ndarray:
    ## TODO
    next_state = self.observation_space.sample()
    return next_state

# We create a filename for the two saved models: the Actor and Critic Models
file_name = "%s_%s" % ("DDPG", "DIET")
print("---------------------------------------")
print("Settings: %s" % file_name)
print("---------------------------------------")

# We get the necessary information on the states and actions in the chosen environment
state_dim = 6
action_dim = 1
max_action = 21

# We create the policy network (the Actor model)
policy = DDPG(state_dim, action_dim, max_action)

# We create the Experience Replay memory
replay_buffer = ReplayBuffer()


os.chdir(r'C:\Users\Harold\Desktop\ENAC-Semester-Project\DIET_Controller')

TRAIN_PATH = "./Training_Data/01032022/Ep"
MODEL_PATH = "./pytorch_models/01032022"

## At each episode, we need to reset the EnergyPlus environment
## Before doing anything, we train the model on what is stored in the replay buffer

for sim_num in range(num_total_episodes):



    if sim_num != 0:
        print("Total timesteps: {} Episode Num: {} Reward: {}".format(numsteps, sim_num, np.sum(reward.flatten())))
        iterations = (numsteps - 1) * sim_num
        policy.train(replay_buffer,1000 , batch_size, discount, tau, policy_noise, noise_clip)

        if sim_num > 1:
            policy.save(file_name, directory=MODEL_PATH)  # Change the folder name here

    while simtime < timestop:

        if sim_num < num_random_episodes:
            action[index] = round(random.uniform(min_temp, max_temp), 1)  # Choosing random values between 12 and 30 deg

        else:  # After 1 episode, we switch to the model
            action_arr = policy.select_action(obs)
            #print(f"Current obs {obs}")
            #print(f"Selected action is {action_arr[0]}")
            # If the explore_noise parameter is not 0, we add noise to the action and we clip it
            if expl_noise != 0:
                action_arr = (action_arr + np.random.normal(0, expl_noise, size=1)).clip(min_temp, max_temp)
                action[index] = action_arr[0]

        
    os.makedirs(TRAIN_PATH+str(sim_num+1), exist_ok=True)
    # Writing to .csv files to save the data from the episode
    np.savetxt(TRAIN_PATH+str(sim_num+1)+"/state.csv", state[:-1, :], delimiter=",")
    np.savetxt(TRAIN_PATH+str(sim_num+1)+"/next_state.csv", state[1:, :], delimiter=",")
    np.savetxt(TRAIN_PATH+str(sim_num+1)+"/action.csv", action[1:, :], delimiter=",")
    np.savetxt(TRAIN_PATH+str(sim_num+1)+"/reward.csv", reward[1:, :], delimiter=",")
    np.savetxt(TRAIN_PATH+str(sim_num+1)+"/pmv.csv", pmv[1:, :], delimiter=",")

    # Plotting the summary of simulation
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}],
                               [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": False}]])

    # Add traces
    fig.add_trace(go.Scatter(name='Tair(state)', x=t, y=tair.flatten(), mode='lines', line=dict(width=1, color='cyan')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(name='Tair_avg', x=t, y=pd.Series(tair.flatten()).rolling(window=24).mean(), mode='lines',
                  line=dict(width=2, color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(name='Tset(action)', x=t, y=action.flatten(), mode='lines', line=dict(width=1, color='fuchsia')),
                  row=2, col=1)
    fig.add_trace(go.Scatter(name='Tset_avg', x=t, y=pd.Series(action.flatten()).rolling(window=24).mean(), mode='lines',
                  line=dict(width=2, color='purple')), row=2, col=1)
    fig.add_trace(go.Scatter(name='Pmv', x=t, y=pmv.flatten(), mode='lines', line=dict(width=1, color='gold')),
                  row=3, col=1)
    fig.add_trace(go.Scatter(name='Pmv_avg', x=t, y=pd.Series(pmv.flatten()).rolling(window=24).mean(), mode='lines',
                  line=dict(width=2, color='darkorange')), row=3, col=1)
    fig.add_trace(go.Scatter(name='Heating', x=t, y=qheat.flatten(), mode='lines', line=dict(width=1, color='red')),
                  row=4, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(name='Heating_cumulative', x=t, y=np.cumsum(qheat.flatten()), mode='lines',
                  line=dict(width=2, color='darkred')), row=4, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(name='Reward', x=t, y=reward.flatten(), mode='lines', line=dict(width=1, color='lime')),
                  row=5, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(name='Reward_cum', x=t, y=np.cumsum(reward.flatten()), mode='lines',
                  line=dict(width=2, color='darkgreen')), row=5, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(name='Occupancy', x=t, y=occ.flatten(), mode='lines',
                  line=dict(width=1, color='black')), row=6, col=1)

    # Set x-axis title
    fig.update_xaxes(title_text="Timestep (-)", row=6, col=1)

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Tair</b> (°C)", range=[10, 24], row=1, col=1)
    fig.update_yaxes(title_text="<b>Tset</b> (°C)", range=[12, 30], row=2, col=1)
    fig.update_yaxes(title_text="<b>PMV</b> (-)", row=3, col=1)
    fig.update_yaxes(title_text="<b>Heat Power</b> (kJ/hr)", row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="<b>Heat Energy</b> (kJ)", row=4, col=1, secondary_y=True)
    fig.update_yaxes(title_text="<b>Reward</b> (-)", row=5, col=1, range=[-5, 5], secondary_y=False)
    fig.update_yaxes(title_text="<b>Tot Reward</b> (-)", row=5, col=1, secondary_y=True)
    fig.update_yaxes(title_text="<b>Fraction</b> (-)", row=6, col=1)

    fig.update_xaxes(nticks=50)

    fig.update_layout(template='plotly_white', font=dict(family="Courier New, monospace", size=12),
                      legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1))

    pyo.plot(fig, filename=TRAIN_PATH+str(sim_num+1)+"/results.html")

    del model, opts

policy.save(file_name, directory=MODEL_PATH)  # Change the folder name here