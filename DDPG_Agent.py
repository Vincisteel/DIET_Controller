# DDPG Thermal Control Policy Implementation

from typing import Dict, List, Tuple, Any
from Agent import *
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import gym
import random
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
        self.is_test = False

    def from_dict(self, dict_arguments: Dict[str, Any]) -> Agent:
        for k, v in dict_arguments.items():
            setattr(self, k, v)

        return self

    def select_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        next_state, reward, done, info = self.env.step(action)

        return next_state, reward, done, info

    def train(
        self,
        logging_path: str,
        num_episodes: int,
        num_iterations: int,
        log: bool,
        is_test: bool = False,
        num_training_iterations: int = 100,
        num_random_episodes: int = 1,
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

        self.is_test = is_test

        summary_df: pd.DataFrame = pd.DataFrame()

        epsilons = []
        losses = []
        tair = []
        actions = []
        pmv = []
        qheat = []
        rewards = []
        occ = []

        for episode_num in range(num_episodes):

            state = self.env.reset()
            os.chdir(logging_path)

            ## update the model when the replay buffer is filled
            if not (self.is_test) and (episode_num >= num_random_episodes):
                ## train the agent
                self.update_agent(num_training_iterations)

            for i in range(num_iterations):

                if i % 1000 == 0:
                    print(f"Iteration{i}")

                if episode_num < num_random_episodes:
                    actions.append(
                        round(random.uniform(self.env.min_temp, self.env.max_temp), 1)
                    )
                else:
                    action = self.select_action(state)
                    next_state, reward, done, info = self.step(action)

                    ## keeping track of the value we've seen
                    rewards.append(reward)
                    actions.append(action)
                    pmv.append(info["pmv"][0])
                    d = self.env.observation_to_dict(next_state)
                    tair.append(d["Tair"][0])
                    heat = d["Qheat"][0]
                    qheat.append(heat)
                    occ.append(d["Occ"][0])

                    self.replay_buffer.add((state, next_state, action, reward))

                    state = next_state

            lower = episode_num * num_iterations
            upper = (episode_num + 1) * num_iterations
            if log:
                summary_df = logger.plot_and_logging(
                    episode_num,
                    tair[lower:upper],
                    actions[lower:upper],
                    pmv[lower:upper],
                    qheat[lower:upper],
                    rewards[lower:upper],
                    occ[lower:upper],
                    losses[lower:upper],
                    epsilons[lower:upper],
                    self,
                )

        # plot a summary that contatenates all episodes together for a complete overview of the training
        if log and num_episodes > 1:
            summary_df = logger.plot_and_logging(
                episode_num,
                tair,
                actions,
                pmv,
                qheat,
                rewards,
                occ,
                losses,
                epsilons,
                self,
                is_summary=True,
            )

        # self.env.close()

        results_path = logger.RESULT_PATH

        return (results_path, summary_df)

    def test(
        self, logging_path: str, num_episodes: int, num_iterations: int, log: bool
    ) -> Tuple[str, pd.DataFrame]:

        return self.train(
            logging_path=logging_path,
            num_episodes=num_episodes,
            num_iterations=num_iterations,
            log=log,
            num_random_episodes=0,
            is_test=True,
        )

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

    def update_agent(self, num_training_iterations: int):

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
