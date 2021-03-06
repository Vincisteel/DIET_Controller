# DDPG Thermal Control Policy Implementation

# Importing the libraries
import os
import time
import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import subprocess  # to run the TRNSYS simulation
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
        print(xu.size())
        print(x.size(), u.size())
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1


# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2,
              noise_clip=0.5):

        for it in range(iterations):

            # Step 4: We sample a batch of transitions (s, s', a, r) from the memory
            batch_states, batch_next_states, batch_actions, batch_rewards = replay_buffer.sample(
                batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)

            # Step 5: From the next state s', the Actor target plays the next action a'
            next_action = self.actor_target(next_state)
            print(next_action.size())
            break

            # Step 6: We add Gaussian noise to this next action a' and we clamp it in a range of values supported
            # by the environment
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Step 7: The Critic Target take (s', a') as input and return Q-value Qt(s', a') as output
            target_q = self.critic_target(next_state, next_action)

            # Step 8: We get the estimated reward, which is: r' = r + ?? * Qt, where ?? id the discount factor
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


# We make a function that evaluates the policy by calculating its average reward over 3 episodes
def evaluate_policy(eval_episodes=1):
    avg_reward = 0.
    for eval_sim in range(eval_episodes):
        # We reset the state of the environment [Tair, RHair, Tmrt, Vair, Tout, Clo, Met, Occ, Qheat, HOY]
        obs = np.array([20.0, 50.0, 20.0, 0.1, 5.5, 1.0, 1.0, 0.0001, 0.0001, 0.0])

        # Store the first observations in the text file
        state_txt = open(r"C:\Users\achatter\Desktop\PhDResearch\DIET\Controller\py_state.dat", "w")
        state_txt.truncate(0)
        state_txt.write('\t' + str(obs[0]) + '\t' + str(obs[1]) + '\t' + str(obs[2]) + '\t' + str(obs[3]) + '\t' +
                        str(obs[4]) + '\t' + str(obs[5]) + '\t' + str(obs[6]) + '\t' + str(obs[7]) + '\t' +
                        str(obs[8]) + '\n')
        state_txt.close()

        # Erase the data from the previous episodes
        next_state_txt = open(r"C:\Users\achatter\Desktop\PhDResearch\DIET\Controller\py_next_state.dat", "w")
        next_state_txt.truncate(0)
        next_state_txt.close()

        action_txt = open(r"C:\Users\achatter\Desktop\PhDResearch\DIET\Controller\py_action.dat", "w")
        action_txt.truncate(0)
        action_txt.close()

        reward_txt = open(r"C:\Users\achatter\Desktop\PhDResearch\DIET\Controller\py_reward.dat", "w")
        reward_txt.truncate(0)
        reward_txt.close()

        pmv_txt = open(r"C:\Users\achatter\Desktop\PhDResearch\DIET\Controller\py_pmv.dat", "w")
        pmv_txt.truncate(0)
        pmv_txt.close()

        # Running TRNSYS simulation
        subprocess.run([r"C:\TRNSYS18\Exe\TrnEXE64.exe",
                        r"C:\Users\achatter\Desktop\PhDResearch\DIET\Controller\BuildingModel_1day.dck"])

        # Reading the reward from text file and calculating the episode reward
        reward_data = pd.read_csv(
            r"C:\Users\achatter\Desktop\PhDResearch\DIET\Controller\Backup\7\Training\py_reward.dat",
            sep="\s+", usecols=[0], names=[0], skiprows=2)
        avg_reward += reward_data[0].sum

    avg_reward /= eval_episodes
    print("---------------------------------------")
    print("Average Reward over the Evaluation Step: %f" % avg_reward)
    print("---------------------------------------")
    return avg_reward


# We set the parameters
start_sim = 2  # Number of simulations after which the reinforcement learning defines the actions
max_sim = 10  # Total number of Trnsys simulations for training
sim_num = 0
save_models = True  # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.1  # Exploration noise - STD value of exploration Gaussian noise
batch_size = 100  # Size of the batch
discount = 0.99  # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005  # Target network update rate
policy_noise = 0.2  # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5  # Maximum value of the Gaussian noise added to the actions (policy)
comfort_lim = 0.5  # Limiting value of comfort(+/-) on the PMV index
alpha = 0.5  # Adjusting co-efficient for the comfort reward
beta = 1  # Adjusting co-efficient for the energy reward
obs = np.array([20.0, 50.0, 20.0, 0.1, 5.5, 1.0, 1.0, 0.0001, 0.0001])  # Initial state of the trnsys env

# We create a filename for the two saved models: the Actor and Critic Models
file_name = "%s_%s" % ("DDPG", "DIET")
print("---------------------------------------")
print("Settings: %s" % file_name)
print("---------------------------------------")

# We get the necessary information on the states and actions in the chosen environment
state_dim = 9
action_dim = 1
max_action = 21

# We create the policy network (the Actor model)
policy = DDPG(state_dim, action_dim, max_action)

# We create the Experience Replay memory
replay_buffer = ReplayBuffer()

# We define a list where all the evaluation results over 3 episodes are stored
evaluations = []


# Function taken from CBE comfort tool to calculate the pmv value for comfort evaluation
def comfPMV(ta, tr, vel, rh, met, clo, wme=0):
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


def trnsys_sim(tair_in, rh_in, tmrt_in, vair_in, tout_in, clo_in, met_in, occ_in, qheat_in):
    global obs
    # Retrieve values from the parameters and inputs from Trnsys

    # Calculate the values of the new actions
    if sim_num < start_sim:
        action = round(random.uniform(16, 21), 1)  # Choosing random values between 16 and 24 deg
    else:  # After 2 episodes, we switch to the model
        action_arr = policy.select_action(obs)
        # If the explore_noise parameter is not 0, we add noise to the action and we clip it
        if expl_noise != 0:
            action_arr = (action_arr + np.random.normal(0, expl_noise, size=1)).clip(16.0, 21.0)
            action = action_arr[0]

    # The agent performs the action in the environment, then reaches the next state and receives the reward
    new_obs = np.array([tair_in, rh_in, tmrt_in, vair_in, tout_in, clo_in, met_in, occ_in, qheat_in])

    pmv = comfPMV(tair_in, tmrt_in, vair_in, rh_in, met_in, clo_in)

    reward = beta * (1 - (qheat_in/15000)) + alpha * (1 - ((pmv + 0.5) ** 2)) * occ_in

    obs = new_obs

    # Step 6: Return the new values based on which action will be performed in Trnsys
    return action, reward, pmv


def run_training():
    global sim_num
    action_data = pd.DataFrame()
    reward_data = pd.DataFrame()

    for ep_num in range(max_sim):

        sim_num = ep_num
        # If we are not at the very beginning, we start the training process of the model
        if ep_num != 0:
            print("Total simulations: {} Episode Num: {} Reward: {}".format(action_data.size, ep_num + 1,
                                                                            reward_data[0].sum))
            policy.train(replay_buffer, action_data.size, batch_size, discount, tau, policy_noise, noise_clip)

            evaluations.append(evaluate_policy())
            policy.save(file_name, directory="./pytorch_models/latest")
            np.save("./results/%s" % file_name, evaluations)

        # We reset the state of the environment [Tair, RHair, Tmrt, Vair, Tout, Clo, Met, Occ, Qheat, HOY]
        obs = np.array([20.0, 50.0, 20.0, 0.1, 5.5, 1.0, 1.0, 0.0001, 0.0001])

        # Store the first observations in the text file
        state_txt = open(r"C:\Users\achatter\Desktop\PhDResearch\DIET\Controller\py_state.dat", "w")
        state_txt.truncate(0)
        state_txt.write('\t' + str(obs[0]) + '\t' + str(obs[1]) + '\t' + str(obs[2]) + '\t' + str(obs[3]) + '\t' +
                        str(obs[4]) + '\t' + str(obs[5]) + '\t' + str(obs[6]) + '\t' + str(obs[7]) + '\t' +
                        str(obs[8]) + '\n')
        state_txt.close()

        # Erase the data from the previous episodes
        next_state_txt = open(r"C:\Users\achatter\Desktop\PhDResearch\DIET\Controller\py_next_state.dat", "w")
        next_state_txt.truncate(0)
        next_state_txt.close()

        action_txt = open(r"C:\Users\achatter\Desktop\PhDResearch\DIET\Controller\py_action.dat", "w")
        action_txt.truncate(0)
        action_txt.close()

        reward_txt = open(r"C:\Users\achatter\Desktop\PhDResearch\DIET\Controller\py_reward.dat", "w")
        reward_txt.truncate(0)
        reward_txt.close()

        pmv_txt = open(r"C:\Users\achatter\Desktop\PhDResearch\DIET\Controller\py_pmv.dat", "w")
        pmv_txt.truncate(0)
        pmv_txt.close()

        # Running TRNSYS simulation
        subprocess.run([r"C:\TRNSYS18\Exe\TrnEXE64.exe",
                        r"C:\Users\achatter\Desktop\PhDResearch\DIET\Controller\BuildingModel_2day.dck"])

        # Reading from the text files to fill the replay buffer
        state_data = pd.read_csv(
            r"C:\Users\achatter\Desktop\PhDResearch\DIET\Controller\Backup\7\Training\py_state.dat",
            sep="\s+", usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8], names=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            skiprows=2, skipfooter=1)
        next_state_data = pd.read_csv(
            r"C:\Users\achatter\Desktop\PhDResearch\DIET\Controller\Backup\7\Training\py_next_state.dat", sep="\s+",
            usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8], names=[0, 1, 2, 3, 4, 5, 6, 7, 8], skiprows=2)
        action_data = pd.read_csv(
            r"C:\Users\achatter\Desktop\PhDResearch\DIET\Controller\Backup\7\Training\py_action.dat",
            sep="\s+", usecols=[0], names=[0], skiprows=2)
        reward_data = pd.read_csv(
            r"C:\Users\achatter\Desktop\PhDResearch\DIET\Controller\Backup\7\Training\py_reward.dat",
            sep="\s+", usecols=[0], names=[0], skiprows=2)

        for ind in range(action_data.size):
            obs = np.array(
                [state_data[0][ind], state_data[1][ind], state_data[2][ind], state_data[3][ind], state_data[4][ind],
                 state_data[5][ind], state_data[6][ind], state_data[7][ind],
                 state_data[8][ind]])

            new_obs = np.array(
                [next_state_data[0][ind], next_state_data[1][ind], next_state_data[2][ind], next_state_data[3][ind],
                 next_state_data[4][ind], next_state_data[5][ind],
                 next_state_data[6][ind], next_state_data[7][ind], next_state_data[8][ind]])

            action = np.array([action_data[0][ind]])
            reward_arr = np.array([reward_data[0][ind]])

            # Step 5: We store the new transition into the Experience Replay memory (ReplayBuffer)
            replay_buffer.add((obs, new_obs, action, reward_arr))

    # We add the last policy evaluation to our list of evaluations and we save our model
    evaluations.append(evaluate_policy())
    if save_models:
        policy.save("%s" % file_name, directory="./pytorch_models/latest")
    np.save("./results/%s" % file_name, evaluations)




