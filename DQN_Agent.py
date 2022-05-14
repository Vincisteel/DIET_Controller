from typing import Dict, List, Tuple, Any
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

from Agent import *


class DQNAgent(Agent):
    """DQN Agent interacting with environment.
    
    Attribute:
        env : (Environment) custom Environment to interact with TRNSYS
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        epsilon (float): parameter for epsilon greedy policy
        epsilon_decay (float): step size to decrease epsilon
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value of epsilon
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
    """

    def __init__(
        self,
        env: gym.Env,
        memory_size: int = 1000,
        batch_size: int = 32,
        target_update: int = 100,
        epsilon_decay: float = 1 / 20000,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
        inside_dim: int = 128,  ## dimension of the hidden layers of the network
        num_hidden_layers: int = 1,
        seed: int = 778,
        dict_arguments: Dict[
            str, Any
        ] = {},  ## easy way to set arguments when using blackbox optimization
    ):
        """Initialization.
        
        Args:
            env (gym.Env): custom Environment to interact with TRNSYS
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            epsilon_decay (float): step size to decrease epsilon
            lr (float): learning rate
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            gamma (float): discount factor
        """

        self.env = env
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma
        self.seed = seed
        self.inside_dim = inside_dim
        self.num_hidden_layers = num_hidden_layers

        if dict_arguments is not None:
            ## set arguments given in directory
            for k, v in dict_arguments.items():
                setattr(self, k, v)

        ## seeding the agent
        self.seed_agent(self.seed)

        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## dimensions for the network
        obs_dim = self.env.observation_dim
        action_dim = self.env.action_dim

        ## setting up memory
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)

        print(f"Agent Action_dim{ action_dim}")

        # networks: dqn, dqn_target
        self.dqn = Network(
            obs_dim,
            action_dim,
            inside_dim=self.inside_dim,
            num_hidden_layers=self.num_hidden_layers,
        ).to(self.device)
        self.dqn_target = Network(
            obs_dim,
            action_dim,
            inside_dim=self.inside_dim,
            num_hidden_layers=self.num_hidden_layers,
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

    def from_dict(self, dict_arguments: Dict[str, Any]) -> Agent:
        for k, v in dict_arguments.items():
            setattr(self, k, v)

        return self

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = np.random.choice(self.env.action_dim, 1)[0]
        else:
            selected_action = self.dqn(
                torch.FloatTensor(state.T).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take an action and return the response of the env."""
        next_state, reward, done, info = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, done, info

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()

        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(
        self,
        logging_path: str,
        num_iterations=None,
        num_episodes=1,
        log=True,
        is_test=False,
    ) -> Tuple[str, pd.DataFrame]:
        """Train the agent."""
        self.is_test = is_test

        ## check num_iterations
        if num_iterations is None:
            num_iterations = self.env.numsteps

        if num_iterations > self.env.numsteps:
            print(
                f"WARNING: Number of iterations chosen ({num_iterations}) is higher than the number of steps of the environment ({self.env.numsteps}) "
            )
            num_iterations = self.env.numsteps

        ## instantiate logger
        logger = Logger(
            logging_path=logging_path,
            agent_name="DQN_Agent",
            num_episodes=num_episodes,
            num_iterations=num_iterations,
        )

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
            ## chdir back to logging path otherwise then we recall train() mutliple times, the  os.getcwd() will have moved
            os.chdir(logging_path)

            ## to keep track of number of updates and update target network accordingly
            update_cnt = 0

            for i in range(num_iterations):

                action = self.select_action(state)
                next_state, reward, done, info = self.step(action)

                if i % 1000 == 0:
                    print(f"Iteration{i}")

                ## keeping track of the value we've seen
                rewards.append(reward)
                actions.append(self.env.action_to_temp[action])
                pmv.append(info["pmv"][0])
                d = self.env.observation_to_dict(next_state)
                tair.append(d["Tair"][0])
                heat = d["Qheat"][0]
                qheat.append(heat)
                occ.append(d["Occ"][0])

                state = next_state

                # if episode ends
                # if done:
                #    state = self.env.reset()

                # if training is ready
                if not (self.is_test) and (len(self.memory) >= self.batch_size):
                    loss = self.update_model()
                    losses.append(loss)
                    update_cnt += 1

                    # linearly decrease epsilon
                    self.epsilon = max(
                        self.min_epsilon,
                        self.epsilon
                        - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay,
                    )
                    epsilons.append(self.epsilon)

                    # if hard update is needed
                    if not (self.is_test) and (update_cnt % self.target_update == 0):
                        self._target_hard_update()

            ## slicing lower and upper bound
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
        self, logging_path: str, num_iterations=None, num_episodes=1, log=True
    ) -> Tuple[str, pd.DataFrame]:

        return self.train(
            is_test=True,
            logging_path=logging_path,
            num_iterations=num_iterations,
            num_episodes=num_episodes,
            log=log,
        )

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def seed_agent(self, seed):

        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        np.random.seed(seed)

        return

    # Making a save method to save a trained model
    def save(self, filename, directory):
        torch.save(self.dqn.state_dict(), "%s/%s_dqn.pth" % (directory, filename))
        torch.save(
            self.dqn_target.state_dict(), "%s/%s_dqn_target.pth" % (directory, filename)
        )

    # Making a load method to load a pre-trained model
    def load(self, filename, directory):
        self.dqn.load_state_dict(torch.load("%s/%s_dqn.pth" % (directory, filename)))
        self.dqn_target.load_state_dict(
            torch.load("%s/%s_dqn_target.pth" % (directory, filename))
        )

    def log_dict(self) -> Dict[str, Any]:
        log_dict = {
            "is_test": self.is_test,
            "memory_size": self.memory_size,
            "batch_size": self.batch_size,
            "target_update": self.target_update,
            "epsilon_decay": self.epsilon_decay,
            "max_epsilon": self.max_epsilon,
            "min_epsilon": self.min_epsilon,
            "gamma": self.gamma,
            "inside_dim": self.inside_dim,
            "num_hidden_layers": self.num_hidden_layers,
            "seed": self.seed,
        }

        return log_dict


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_dim = obs_dim
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs.reshape(self.obs_buf[self.ptr].shape)
        self.next_obs_buf[self.ptr] = next_obs.reshape(self.obs_buf[self.ptr].shape)
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


class Network(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, inside_dim: int = 128, num_hidden_layers=1
    ):
        """Initialization."""
        super(Network, self).__init__()

        # first layer
        self.layers = nn.ModuleList([nn.Linear(in_dim, inside_dim)])
        self.layers.append(nn.ReLU())

        for i in range(num_hidden_layers):
            self.layers.append(nn.Linear(inside_dim, inside_dim))
            self.layers.append(nn.ReLU())

        # last layer
        self.layers = self.layers.append(nn.Linear(inside_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        for layer in self.layers:
            x = layer(x)

        return x
