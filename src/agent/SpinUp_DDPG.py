from turtle import update
import torch
import torch.nn as nn

from typing import Dict, List, Tuple, Any
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
import os

from Logger import SimpleLogger
from agent.Agent import Agent
from environment.Environment import Environment


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class MLPActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: List[int],
        activation,
        min_action: float,
        max_action: float,
    ):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Sigmoid)
        self.min_action = min_action
        self.max_action = max_action

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.min_action + self.pi(obs) * (self.max_action - self.min_action)


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int], activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        min_action: float,
        max_action: float,
        hidden_sizes: List[int] = [256, 256],
        activation=nn.ReLU,
    ):
        super().__init__()

        # build policy and value functions
        self.pi = MLPActor(
            obs_dim, act_dim, hidden_sizes, activation, min_action, max_action
        )
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def __len__(self) -> int:
        return self.size


class DDPGSpinUp(Agent):
    def __init__(
        self,
        env: Environment,
        seed=0,
        steps_per_epoch=4000,
        epochs=100,
        memory_size=1000,
        gamma=0.99,
        polyak=0.995,
        lr=1e-3,
        batch_size=100,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.0,
        epsilon_decay: float = 1/20000,
        start_steps=10000,
        update_after=1000,
        update_every=50,
        noise_scale=0.1,
        num_test_episodes=10,
        max_ep_len=1000,
    ):
        """
        Deep Deterministic Policy Gradient (DDPG)
        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.
            actor_critic: The constructor method for a PyTorch Module with an ``act`` 
                method, a ``pi`` module, and a ``q`` module. The ``act`` method and
                ``pi`` module should accept batches of observations as inputs,
                and ``q`` should accept a batch of observations and a batch of 
                actions as inputs. When called, these should return:
                ===========  ================  ======================================
                Call         Output Shape      Description
                ===========  ================  ======================================
                ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                               | observation.
                ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                               | given observations.
                ``q``        (batch,)          | Tensor containing the current estimate
                                               | of Q* for the provided observations
                                               | and actions. (Critical: make sure to
                                               | flatten this!)
                ===========  ================  ======================================
            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
                you provided to DDPG.
            seed (int): Seed for random number generators.
            steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
                for the agent and the environment in each epoch.
            epochs (int): Number of epochs to run and train agent.
            replay_size (int): Maximum length of replay buffer.
            gamma (float): Discount factor. (Always between 0 and 1.)
            polyak (float): Interpolation factor in polyak averaging for target 
                networks. Target networks are updated towards main networks 
                according to:
                .. math:: \\theta_{\\text{targ}} \\leftarrow 
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta
                where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
                close to 1.)
            pi_lr (float): Learning rate for policy.
            q_lr (float): Learning rate for Q-networks.
            batch_size (int): Minibatch size for SGD.
            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.
            update_after (int): Number of env interactions to collect before
                starting to do gradient descent updates. Ensures replay buffer
                is full enough for useful updates.
            update_every (int): Number of env interactions that should elapse
                between gradient descent updates. Note: Regardless of how long 
                you wait between updates, the ratio of env steps to gradient steps 
                is locked to 1.
            act_noise (float): Stddev for Gaussian exploration noise added to 
                policy at training time. (At test time, no noise is added.)
            num_test_episodes (int): Number of episodes to test the deterministic
                policy at the end of each epoch.
            max_ep_len (int): Maximum length of trajectory / episode / rollout.
            logger_kwargs (dict): Keyword args for EpochLogger.
            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.
        """

        super().__init__()

        self.env = env
        self.polyak = polyak
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.noise_scale = noise_scale
        self.memory_size = memory_size
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon

        self.seed_agent(seed)
        self.is_test = False

        # Create actor-critic module and target networks
        self.ac = MLPActorCritic(
            obs_dim=env.observation_dim,
            act_dim=env.action_dim,
            min_action=self.env.min_temp,
            max_action=self.env.max_temp,
        )
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # Experience buffer
        self.replay_buffer = ReplayBuffer(
            obs_dim=env.observation_dim, act_dim=env.action_dim, size=self.memory_size
        )

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.lr)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=self.lr)
        
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
            "Epsilon": {"secondary_y": None, "range": None, "unit": "(-)",},
            "Loss": {"secondary_y": None, "range": None, "unit": "(-)",},
        }


    def compute_loss_q(self, data):
        obs, action, r, o2 = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
        )
        q = self.ac.q(obs, action)
        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            backup = r + self.gamma * q_pi_targ
        # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()

        return loss_q

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(self, data):
        obs = data["obs"]
        q_pi = self.ac.q(obs, self.ac.pi(obs))
        return -q_pi.mean()

    # Set up model saving

    def update(self, data):
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()
        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in self.ac.q.parameters():
            p.requires_grad = False
        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()
        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.ac.q.parameters():
            p.requires_grad = True

        self.update_target()


    def update_target(self):
        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)


    def select_action(self, obs:np.ndarray) -> np.ndarray:

        #epsilon greedy policy
        if self.epsilon > np.random.random():

        a = self.ac.act(torch.as_tensor(obs, dtype=torch.float32))
        if not (self.is_test):
            a += self.noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, self.env.min_temp, self.env.max_temp)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:

        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1

    def train(
        self,
        logging_path: str,
        num_episodes: int,
        num_iterations: int,
        log: bool,
        is_test: bool = False,
    ) -> Tuple[str, pd.DataFrame]:

        self.is_test = is_test

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

            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy (with some noise, via act_noise).
            if t > start_steps:
                a = get_action(o, act_noise)
            else:
                a = env.action_space.sample()
            # Step the env
            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1
            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == max_ep_len else d
            # Store experience to replay buffer
            replay_buffer.store(o, a, r, o2, d)
            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2
            # End of trajectory handling
            if d or (ep_len == max_ep_len):
                o, ep_ret, ep_len = env.reset(), 0, 0
            # Update handling
            if t >= update_after and t % update_every == 0:
                for _ in range(update_every):
                    batch = self.replay_buffer.sample_batch(batch_size)
                    self.update(data=batch)
            # End of epoch handling
            if (t + 1) % steps_per_epoch == 0:
                epoch = (t + 1) // steps_per_epoch
                # Test the performance of the deterministic version of the agent.
                test_agent()

