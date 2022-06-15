# DDPG Thermal Control Policy Implementation

from typing import Dict, List, Tuple, Any
from agent.Agent import Agent
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

from logger.SimpleLogger import SimpleLogger
from environment.Environment import Environment


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_dim = obs_dim
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray,
    ):
        self.obs_buf[self.ptr] = obs.reshape(self.obs_buf[self.ptr].shape)
        self.next_obs_buf[self.ptr] = next_obs.reshape(self.obs_buf[self.ptr].shape)
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return (
            self.obs_buf[idxs],
            self.next_obs_buf[idxs],
            self.acts_buf[idxs],
            self.rews_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


# Building a neural network for the actor model and a neural network for the actor target
class Actor(nn.Module):
    def __init__(self, observation_dim, action_dim, min_action, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(observation_dim, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, action_dim)
        self.min_action = min_action
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.min_action + torch.sigmoid(self.layer_3(x)) * (
            self.max_action - self.min_action
        )
        return x


# Building a neural network for the critic model and a neural network for the critic target
class Critic(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(observation_dim + action_dim, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, action_dim)

    def forward(self, x, u):
        xu = torch.cat([x, u.reshape((x.shape[0], 1))], dim=1)
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
        env: Environment,
        memory_size: int = 1000,
        batch_size: int = 128,
        discount: float = 0.99,
        tau: float = 0.05,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        lr: float = 1e-3,
        num_training_iterations: int = 100,
        num_random_episodes: int = 0,
        seed: int = 778,
    ):

        # seeding the agent
        self.seed_agent(seed)

        self.env = env
        self.memory_size = memory_size
        self.num_training_iterations = num_training_iterations
        self.num_random_episodes = num_random_episodes

        # Selecting the device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## setting up actors and neural networks components
        self.actor = Actor(
            self.env.observation_dim,
            self.env.action_dim,
            min_action=self.env.min_temp,
            max_action=self.env.max_temp,
        ).to(self.device)
        self.actor_target = Actor(
            self.env.observation_dim,
            self.env.action_dim,
            min_action=self.env.min_temp,
            max_action=self.env.max_temp,
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic = Critic(self.env.observation_dim, self.env.action_dim).to(
            self.device
        )
        self.critic_target = Critic(self.env.observation_dim, self.env.action_dim).to(
            self.device
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        ## setting up agent hyperparameters
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.lr = lr

        self.opts = {
            "Tair": {"secondary_y": None, "range": [10, 24], "unit": "(°C)",},
            "Tset": {
                "secondary_y": "moving_average",
                "range": [14, 22],
                "unit": "(°C)",
            },
            "PMV": {"secondary_y": None, "range": [-3, 3], "unit": "(-)",},
            "Heating": {"secondary_y": "cumulative", "range": None, "unit": "(kJ)",},
            "Reward": {"secondary_y": "cumulative", "range": [-5, 5], "unit": "(-)",},
            "Occ": {"secondary_y": None, "range": None, "unit": "(-)",},
        }

        self.replay_buffer = ReplayBuffer(
            batch_size=self.batch_size,
            size=self.memory_size,
            obs_dim=self.env.observation_dim,
        )
        self.is_test = False

    def reset(self) -> Agent:
        self.seed_agent(self.seed)

        self.actor = Actor(
            self.env.observation_dim,
            self.env.action_dim,
            min_action=self.env.min_temp,
            max_action=self.env.max_temp,
        ).to(self.device)
        self.actor_target = Actor(
            self.env.observation_dim,
            self.env.action_dim,
            min_action=self.env.min_temp,
            max_action=self.env.max_temp,
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic = Critic(self.env.observation_dim, self.env.action_dim).to(
            self.device
        )
        self.critic_target = Critic(self.env.observation_dim, self.env.action_dim).to(
            self.device
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.replay_buffer = ReplayBuffer(
            batch_size=self.batch_size,
            size=self.memory_size,
            obs_dim=self.env.observation_dim,
        )
        self.is_test = False

        return self

    def select_action(self, state):
        with torch.no_grad():
            state = torch.Tensor(state.reshape(1, -1)).to(self.device)
            action = self.actor(state).cpu().data.numpy().flatten()[0]
            return np.clip(action, self.env.min_temp, self.env.max_temp)

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
    ) -> Tuple[str, pd.DataFrame]:

        self.is_test = is_test
        ## check num_iterations
        if num_iterations is None:
            num_iterations = self.env.numsteps

        if num_iterations > self.env.numsteps:
            print(
                f"WARNING: Number of iterations chosen ({num_iterations}) is higher than the number of steps of the environment ({self.env.numsteps}) "
            )
            num_iterations = self.env.numsteps

        logger = SimpleLogger(
            logging_path=logging_path,
            agent_name="DDPG_Agent",
            num_episodes=num_episodes,
            num_iterations=num_iterations,
        )

        summary_df: pd.DataFrame = pd.DataFrame()

        tair = []
        actions = []
        pmv = []
        qheat = []
        rewards = []
        occ = []

        for episode_num in range(num_episodes):

            state = self.env.reset()
            # need to chdir back to logging_path at each episode because calling env.reset() calls chdir() too
            os.chdir(logging_path)

            ## update the model when the replay buffer is filled
            num_rand = 0 if self.is_test else self.num_random_episodes
            if not (self.is_test) and ((episode_num >= num_rand) and episode_num >= 1):
                ## train the agent
                self.update_agent()

            for i in range(num_iterations):

                if i % 1000 == 0:
                    print(f"Iteration{i}")

                if episode_num < num_rand:
                    action = round(
                        random.uniform(self.env.min_temp, self.env.max_temp), 1
                    )
                else:
                    action = self.select_action(state)

                    if i % 1000 == 0:
                        print("ACTION SELECTED", action)

                next_state, reward, done, info = self.step(action)

                # print("STATE", state, "NEXT_STATE", next_state)

                ## keeping track of the value we've seen
                rewards.append(reward)
                actions.append(action)
                pmv.append(info["pmv"][0])
                d = self.env.observation_to_dict(next_state)
                tair.append(d["Tair"][0])
                heat = d["Qheat"][0]
                qheat.append(heat)
                occ.append(d["Occ"][0])
                self.replay_buffer.store(
                    obs=state, next_obs=next_state, act=np.array(action), rew=reward
                )

                state = next_state

            lower = episode_num * num_iterations
            upper = (episode_num + 1) * num_iterations
            summary_df = pd.DataFrame(
                {
                    "Tair": tair[lower:upper],
                    "Tset": actions[lower:upper],
                    "PMV": pmv[lower:upper],
                    "Heating": qheat[lower:upper],
                    "Reward": rewards[lower:upper],
                    "Occ": occ[lower:upper],
                }
            )
            summary_df["Reward"] = summary_df["Reward"].apply(lambda x: float(x[0]))
            if log:
                logger.plot_and_logging(
                    summary_df=summary_df,
                    agent=self,
                    episode_num=episode_num,
                    is_summary=False,
                    opts=self.opts,
                )

        summary_df = pd.DataFrame(
            {
                "Tair": tair,
                "Tset": actions,
                "PMV": pmv,
                "Heating": qheat,
                "Reward": rewards,
                "Occ": occ,
            }
        )

        summary_df["Reward"] = summary_df["Reward"].apply(lambda x: float(x[0]))

        # plot a summary that contatenates all episodes together for a complete overview of the training
        if log and num_episodes > 1:
            logger.plot_and_logging(
                summary_df=summary_df,
                agent=self,
                episode_num=num_episodes,
                is_summary=True,
                opts=self.opts,
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

    def log_dict(self) -> Dict[str, Any]:

        log_dict = {
            "is_test": self.is_test,
            "memory_size": self.memory_size,
            "batch_size": self.batch_size,
            "discount": self.discount,
            "tau": self.tau,
            "policy_noise": self.policy_noise,
            "noise_clip": self.noise_clip,
            "lr": self.lr,
            "num_training_iterations": self.num_training_iterations,
            "num_random_episodes": self.num_random_episodes,
            "seed": self.seed,
        }

        return log_dict

    def seed_agent(self, seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        np.random.seed(seed)
        random.seed(seed)
        return

    def update_agent(self):

        for it in range(self.num_training_iterations):

            # Step 4: We sample a batch of transitions (s, s', a, r) from the memory
            (
                batch_states,
                batch_next_states,
                batch_actions,
                batch_rewards,
            ) = self.replay_buffer.sample_batch()

            state = torch.Tensor(batch_states).to(self.device)
            next_state = torch.Tensor(batch_next_states).to(self.device)
            # normalize
            next_state = (next_state - next_state.mean()) / next_state.std()
            action = torch.Tensor(batch_actions).to(self.device)
            reward = torch.Tensor(batch_rewards).to(self.device)

            # Step 5: From the next state s', the Actor target plays the next action a'
            next_action = self.actor_target(next_state)

            # Step 6: We add Gaussian noise to this next action a' and we clamp it in a range of values supported
            # by the environment
            noise = (
                torch.Tensor(batch_actions)
                .data.normal_(0, self.policy_noise)
                .to(self.device)
            )
            noise = noise.clamp(-self.noise_clip, self.noise_clip).reshape(
                (self.batch_size, 1)
            )
            next_action = (next_action + noise).clamp(
                self.env.min_temp, self.env.max_temp
            )

            # Step 7: The Critic Target take (s', a') as input and return Q-value Qt(s', a') as output
            with torch.no_grad():
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
            ## CHECK HERE
            temporary = self.critic.forward(state, self.actor(state))
            actor_loss = - temporary.mean()/temporary.std()  ## temporary.std()
            #print(
            #    f"MEAN: {temporary.mean()}, STD: {temporary.std()} ACTOR LOSS:{actor_loss}"
            #)
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
