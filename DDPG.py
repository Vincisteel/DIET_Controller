# DDPG Thermal Control Policy Implementation

# Importing the libraries
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


# Initializing the Experience Replay Memory
class ReplayBuffer(object):

    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards = [], [], [], []
        for i in ind:
            state, next_state, action, reward = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
        return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(
            batch_rewards).reshape(-1, 1)


# Building a neural network for the actor model and a neural network for the actor target
class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x


# Building a neural network for the critic model and a neural network for the critic target
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        #print(xu.size())
        #print(x.size(), u.size())
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1


# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


# Building the whole DDPG Training Process into a class
class DDPG(object):

    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.max_action = max_action

    def select_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

        # Learning Process for the DDPG Algorithm

    def train(self, replay_buffer, iterations, batch_size=128, discount=0.99, tau=0.05, policy_noise=0.2,
              noise_clip=0.5):

        for it in range(iterations):

            

            # Step 4: We sample a batch of transitions (s, s', a, r) from the memory
            batch_states, batch_next_states, batch_actions, batch_rewards = replay_buffer.sample(
                batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device) 
            next_state = (next_state - next_state.mean())/next_state.std()
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)

            # Step 5: From the next state s', the Actor target plays the next action a'
            next_action = self.actor_target(next_state)
            #print(next_action.size())

            # Step 6: We add Gaussian noise to this next action a' and we clamp it in a range of values supported
            # by the environment
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(12, self.max_action)

            # Step 7: The Critic Target take (s', a') as input and return Q-value Qt(s', a') as output
            target_q = self.critic_target(next_state, next_action)
            
            if it%100 == 0:
                print(f"Training iterations {it}")

            # Step 8: We get the estimated reward, which is: r' = r + γ * Qt, where γ id the discount factor
            target_q = reward + (discount * target_q).detach()

            # Step 9: The Critic models take (s, a) as input and return Q-value Q(s, a) as output
            current_q = self.critic(state, action)

            # Step 10: We compute the loss coming from the Critic model: Critic Loss = MSE_Loss(Q(s,a), Qt)
            critic_loss = F.mse_loss(current_q, target_q)

            # Step 11: We back propagate this Critic loss and update the parameters of the Critic model with a SGD
            # optimizer
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Step 12: We update our Actor model by performing gradient ascent on the output of the first Critic model
            actor_loss = -self.critic.forward(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Step 13: We update the weights of the Actor target by polyak averaging
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            # Step 14: We update the weights of the Critic target by polyak averaging
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # Making a save method to save a trained model
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    # Making a load method to load a pre-trained model
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))


# We set the parameters
save_models = True  # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.1  # Exploration noise - STD value of exploration Gaussian noise
batch_size = 128  # Size of the batch
discount = 0.99  # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.05  # Target network update rate
policy_noise = 0.2  # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5  # Maximum value of the Gaussian noise added to the actions (policy)
alpha = 3  # Adjusting co-efficient for the comfort reward
beta = 1  # Adjusting co-efficient for the energy reward
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


# Function taken from CBE comfort tool to calculate the pmv value for comfort evaluation
def comfPMV(ta, tr, rh, vel=0.1, met=1.1, clo=1, wme=0):
    pa = rh * 10 * math.exp(16.6536 - 4030.183 / (ta + 235))

    icl = 0.155 * clo  # thermal insulation of the clothing in M2K/W
    m = met * 58.15  # metabolic rate in W/M2
    w = wme * 58.15  # external work in W/M2
    mw = m - w  # internal heat production in the human body
    if icl <= 0.078:
        fcl = 1 + (1.29 * icl)
    else:
        fcl = 1.05 + (0.645 * icl)

    # heat transfer coefficient by forced convection
    hcf = 12.1 * math.sqrt(vel)
    taa = ta + 273
    tra = tr + 273
    # we have verified that using the equation below or this tcla = taa + (35.5 - ta) / (3.5 * (6.45 * icl + .1))
    # does not affect the PMV value
    tcla = taa + (35.5 - ta) / (3.5 * icl + 0.1)

    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100
    p4 = p1 * taa
    p5 = (308.7 - 0.028 * mw) + (p2 * math.pow(tra / 100.0, 4))
    xn = tcla / 100
    xf = tcla / 50
    eps = 0.00015

    n = 0
    while abs(xn - xf) > eps:
        xf = (xf + xn) / 2
        hcn = 2.38 * math.pow(abs(100.0 * xf - taa), 0.25)
        if hcf > hcn:
            hc = hcf
        else:
            hc = hcn
        xn = (p5 + p4 * hc - p2 * math.pow(xf, 4)) / (100 + p3 * hc)
        n += 1
        if n > 150:
            print('Max iterations exceeded')
            return 1  # fixme should not return 1 but instead PMV=999 as per ashrae standard

    tcl = 100 * xn - 273

    # heat loss diff. through skin
    hl1 = 3.05 * 0.001 * (5733 - (6.99 * mw) - pa)
    # heat loss by sweating
    if mw > 58.15:
        hl2 = 0.42 * (mw - 58.15)
    else:
        hl2 = 0
    # latent respiration heat loss
    hl3 = 1.7 * 0.00001 * m * (5867 - pa)
    # dry respiration heat loss
    hl4 = 0.0014 * m * (34 - ta)
    # heat loss by radiation
    hl5 = 3.96 * fcl * (math.pow(xn, 4) - math.pow(tra / 100.0, 4))
    # heat loss by convection
    hl6 = fcl * hc * (tcl - ta)

    ts = 0.303 * math.exp(-0.036 * m) + 0.028
    pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    ppd = 100.0 - 95.0 * math.exp(-0.03353 * pow(pmv, 4.0) - 0.2179 * pow(pmv, 2.0))

    return pmv


os.chdir(r'C:\Users\Harold\Desktop\ENAC-Semester-Project\DIET_Controller')

TRAIN_PATH = "./Training_Data/01032022/Ep"
MODEL_PATH = "./pytorch_models/01032022"

for sim_num in range(num_total_episodes):

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

        model.set('Thsetpoint_diet', action[index])
        res = model.do_step(current_t=simtime, step_size=secondstep, new_step=True)
        inputcheck_heating[index] = model.get('Thsetpoint_diet')
        tair[index], rh[index], tmrt[index], tout[index], qheat[index], occ[index], inputcheck_heating[index] = model.get(['Tair', 'RH', 'Tmrt', 'Tout', 'Qheat', 'Occ', 'Thsetpoint_diet'])
        state[index][0], state[index][1], state[index][2], state[index][3], state[index][4], state[index][5] = tair[index], rh[index], tmrt[index], tout[index], occ[index], qheat[index]
        pmv[index] = comfPMV(tair[index], tmrt[index], rh[index])
        reward[index] = beta * (1 - (qheat[index]/(800*1000))) + alpha * (1 - abs(pmv[index] + 0.5)) * occ[index]   # * int(bool(occ[index]))

        if index == 0:
            obs = np.array([tair[index], rh[index], tmrt[index], tout[index], occ[index], qheat[index]]).flatten()

        else:
            new_obs = np.array([tair[index], rh[index], tmrt[index], tout[index], occ[index], qheat[index]]).flatten()
            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            replay_buffer.add((obs, new_obs, action[index], reward[index]))
            obs = new_obs

        simtime += secondstep
        index += 1
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