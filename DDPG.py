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

from utils import *


DATA_PATH= "C:\\Users\\Harold\\Desktop\\ENAC-Semester-Project\\DIET_Controller\\"

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

            # Step 6: We add Gaussian noise to this next action a' and we clamp it in a range of values supported
            # by the environment
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Step 7: The Critic Target take (s', a') as input and return Q-value Qt(s', a') as output
            target_q = self.critic_target(next_state, next_action)

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


# We make a function that evaluates the policy by calculating its average reward over 3 episodes
def evaluate_policy(eval_episodes=1):
    avg_reward = 0.
    for eval_sim in range(eval_episodes):
        # We reset the state of the environment [Tair, RHair, Tmrt, Vair, Tout, Clo, Met, Occ, Qheat, HOY]
        obs = np.array([20.0, 50.0, 20.0, 0.1, 5.5, 1.0, 1.0, 0.0001, 0.0001, 0.0])

        # Store the first observations in the text file
        state_txt = open(DATA_PATH + "py_state.dat", "w")
        state_txt.truncate(0)
        state_txt.write('\t' + str(obs[0]) + '\t' + str(obs[1]) + '\t' + str(obs[2]) + '\t' + str(obs[3]) + '\t' +
                        str(obs[4]) + '\t' + str(obs[5]) + '\t' + str(obs[6]) + '\t' + str(obs[7]) + '\t' +
                        str(obs[8]) + '\n')
        state_txt.close()

        # Erase the data from the previous episodes
        next_state_txt = open(DATA_PATH + "py_next_state.dat", "w")
        next_state_txt.truncate(0)
        next_state_txt.close()

        action_txt = open(DATA_PATH + "py_action.dat", "w")
        action_txt.truncate(0)
        action_txt.close()

        reward_txt = open(DATA_PATH + "py_reward.dat", "w")
        reward_txt.truncate(0)
        reward_txt.close()

        pmv_txt = open(DATA_PATH + "py_pmv.dat", "w")
        pmv_txt.truncate(0)
        pmv_txt.close()

        # Running TRNSYS simulation
        subprocess.run([r"C:\TRNSYS18\Exe\TrnEXE64.exe",
                        DATA_PATH + "BuildingModel_1day.dck"])

        # Reading the reward from text file and calculating the episode reward
        reward_data = pd.read_csv(
            DATA_PATH + r"Backup\7\Training\py_reward.dat",
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
        obs = [20.0, 50.0, 20.0, 0.1, 5.5, 1.0, 1.0, 0.0001, 0.0001]

        # Store the first observations in the text file
        state_txt = open(DATA_PATH + "py_state.dat", "w")
        state_txt.truncate(0)
        state_txt.write('\t' + str(obs[0]) + '\t' + str(obs[1]) + '\t' + str(obs[2]) + '\t' + str(obs[3]) + '\t' +
                        str(obs[4]) + '\t' + str(obs[5]) + '\t' + str(obs[6]) + '\t' + str(obs[7]) + '\t' +
                        str(obs[8]) + '\n')

        state_txt.close()

        # Erase the data from the previous episodes
        next_state_txt = open(DATA_PATH + "py_next_state.dat", "w")
        next_state_txt.truncate(0)
        next_state_txt.close()

        action_txt = open(DATA_PATH + "py_action.dat", "w")
        action_txt.truncate(0)
        action_txt.close()

        reward_txt = open(DATA_PATH + "py_reward.dat", "w")
        reward_txt.truncate(0)
        reward_txt.close()

        pmv_txt = open(DATA_PATH + "py_pmv.dat", "w")
        pmv_txt.truncate(0)
        pmv_txt.close()

        # Running TRNSYS simulation
        subprocess.run([r"C:\TRNSYS18\Exe\TrnEXE64.exe",
                        DATA_PATH + "BuildingModel_2day.dck"])

        # Reading from the text files to fill the replay buffer
        state_data = pd.read_csv(
            DATA_PATH + r"Backup\7\Training\py_state.dat",
            sep="\s+", usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8], names=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            skiprows=2, skipfooter=1)
        next_state_data = pd.read_csv(
            DATA_PATH + r"Backup\7\Training\py_next_state.dat", sep="\s+",
            usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8], names=[0, 1, 2, 3, 4, 5, 6, 7, 8], skiprows=2)
        action_data = pd.read_csv(
            DATA_PATH + r"Backup\7\Training\py_action.dat",
            sep="\s+", usecols=[0], names=[0], skiprows=2)
        reward_data = pd.read_csv(
            DATA_PATH + r"Backup\7\Training\py_reward.dat",
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




