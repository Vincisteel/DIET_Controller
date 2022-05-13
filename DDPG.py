# DDPG Thermal Control Policy Implementation

from typing import Dict, List, Tuple, Any
from paramiko import Agent
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import gym
import envs

import os
import pandas as pd

from Logger import *


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
        return (
            np.array(batch_states),
            np.array(batch_next_states),
            np.array(batch_actions),
            np.array(batch_rewards).reshape(-1, 1),
        )


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
        # print(xu.size())
        # print(x.size(), u.size())
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1


# Building the whole DDPG Training Process into a class
class DDPG_Agent(Agent):
    def __init__(
        self,
        env: gym.Env,
        state_dim: int,
        action_dim: int,
        max_action: float,
        batch_size: int = 128,
        discount: float = 0.99,
        tau: float = 0.05,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
    ):

        self.env = env

        # Selecting the device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## setting up actors and neural networks components
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        ## setting up agent hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

        self.replay_buffer = ReplayBuffer()

    def select_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

        # Learning Process for the DDPG Algorithm


    def update_agent(self,num_training_iterations:int):

        for it in range(num_training_iterations):

            # Step 4: We sample a batch of transitions (s, s', a, r) from the memory
            (
                batch_states,
                batch_next_states,
                batch_actions,
                batch_rewards,
            ) = self.replay_buffer.sample(self.batch_size)

            state = torch.Tensor(batch_states).to(self.device)
            next_state = torch.Tensor(batch_next_states).to(self.device)
            # normalize
            next_state = (next_state - next_state.mean()) / next_state.std()
            action = torch.Tensor(batch_actions).to(self.device)
            reward = torch.Tensor(batch_rewards).to(self.device)

            # Step 5: From the next state s', the Actor target plays the next action a'
            next_action = self.actor_target(next_state)
            # print(next_action.size())

            # Step 6: We add Gaussian noise to this next action a' and we clamp it in a range of values supported
            # by the environment
            noise = (
                torch.Tensor(batch_actions)
                .data.normal_(0, self.policy_noise)
                .to(self.device)
            )
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(12, self.max_action)

            # Step 7: The Critic Target take (s', a') as input and return Q-value Qt(s', a') as output
            target_q = self.critic_target(next_state, next_action)

            if it % 100 == 0:
                print(f"Training iterations {it}")

            # Step 8: We get the estimated reward, which is: r' = r + γ * Qt, where γ id the discount factor
            target_q = reward + (self.discount * target_q).detach()

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
            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            # Step 14: We update the weights of the Critic target by polyak averaging
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )


    def train(
        self,
        logging_path: str,
        num_episodes: int,
        num_iterations: int,
        log: bool,
        num_training_iterations: int = 100,
        num_random_episodes: int = 1
    ) -> Tuple[str, pd.DataFrame]:

        ## check num_iterations
        if num_iterations is None:
            num_iterations = self.env.numsteps

        if num_iterations > self.env.numsteps:
            print(
                f"WARNING: Number of iterations chosen ({num_iterations}) is higher than the number of steps of the environment ({self.env.numsteps}) "
            )
            num_iterations = self.env.numsteps

        logger = Logger(
            logging_path=logging_path,
            agent_name="DDPG_Agent",
            num_episodes=num_episodes,
            num_iterations=num_iterations,
        )

        for episode_num in range(num_episodes):

            if episode_num < num_random_episodes:
                ## do random things
            else:
                # do step by step





        

    # Making a save method to save a trained model
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    # Making a load method to load a pre-trained model
    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )


# We get the necessary information on the states and actions in the chosen environment
state_dim = 6
action_dim = 1
max_action = 21

num_random_episodes = 1
num_total_episodes = 3

# We create the policy network (the Actor model)
policy = DDPG_Agent(state_dim, action_dim, max_action)


TRAIN_PATH = "../Training_Data/01032022/Ep"
MODEL_PATH = "../pytorch_models/01032022"

for sim_num in range(num_total_episodes):

    model = load_fmu(modelname + ".fmu")
    opts = model.simulate_options()  # Get the default options
    opts["ncp"] = numsteps  # Specifies the number of timesteps
    opts["initialize"] = False
    simtime = 0
    model.initialize(simtime, timestop)
    index = 0
    t = np.linspace(0.0, numsteps - 1, numsteps)
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
        print(
            "Total timesteps: {} Episode Num: {} Reward: {}".format(
                numsteps, sim_num, np.sum(reward.flatten())
            )
        )
        iterations = (numsteps - 1) * sim_num
        policy.train(
            replay_buffer, 1000, batch_size, discount, tau, policy_noise, noise_clip
        )

        if sim_num > 1:
            policy.save(file_name, directory=MODEL_PATH)  # Change the folder name here

    while simtime < timestop:

        if sim_num < num_random_episodes:
            action[index] = round(
                random.uniform(self.env.min_temp, self.env.max_temp), 1
            )  # Choosing random values between 12 and 30 deg

        else:  # After 1 episode, we switch to the model
            action_arr = policy.select_action(obs)
            # print(f"Current obs {obs}")
            # print(f"Selected action is {action_arr[0]}")
            # If the explore_noise parameter is not 0, we add noise to the action and we clip it
            if expl_noise != 0:
                action_arr = (
                    action_arr + np.random.normal(0, expl_noise, size=1)
                ).clip(self.env.min_temp, self.env.max_temp)
                action[index] = action_arr[0]

        model.set("Thsetpoint_diet", action[index])
        res = model.do_step(current_t=simtime, step_size=secondstep, new_step=True)
        inputcheck_heating[index] = model.get("Thsetpoint_diet")

        ## keeping track of the value we've seeen
        (
            tair[index],
            rh[index],
            tmrt[index],
            tout[index],
            qheat[index],
            occ[index],
            inputcheck_heating[index],
        ) = model.get(["Tair", "RH", "Tmrt", "Tout", "Qheat", "Occ", "Thsetpoint_diet"])

        ## putting them together
        (
            state[index][0],
            state[index][1],
            state[index][2],
            state[index][3],
            state[index][4],
            state[index][5],
        ) = (tair[index], rh[index], tmrt[index], tout[index], occ[index], qheat[index])
        pmv[index] = comfPMV(tair[index], tmrt[index], rh[index])

        ## computing reward
        reward[index] = (
            beta * (1 - (qheat[index] / (800 * 1000)))
            + alpha * (1 - abs(pmv[index] + 0.5)) * occ[index]
        )  # * int(bool(occ[index]))

        if index == 0:
            obs = np.array(
                [
                    tair[index],
                    rh[index],
                    tmrt[index],
                    tout[index],
                    occ[index],
                    qheat[index],
                ]
            ).flatten()

        else:
            new_obs = np.array(
                [
                    tair[index],
                    rh[index],
                    tmrt[index],
                    tout[index],
                    occ[index],
                    qheat[index],
                ]
            ).flatten()

            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            replay_buffer.add((obs, new_obs, action[index], reward[index]))

            obs = new_obs

        simtime += secondstep
        index += 1
