from msilib.schema import Error
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
import torch.nn.functional as F

from Logger import SimpleLogger
from agent.Agent import Agent
from environment.Environment import Environment


class DDPGSpinUp(Agent):
    def __init__(
        self,
        env: Environment,
        seed=0,
        memory_size=1000,
        gamma=0.99,
        polyak=0.995,
        lr=1e-3,
        batch_size=128,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.0,
        epsilon_decay: float = 1 / 20000,
        actor_update: int = 10,
        target_update: int = 100,
        inside_dim: int = 256,  ## dimension of the hidden layers of the network
        num_hidden_layers: int = 2,
        noise_scale=0.1,
    ):
        """
        Deep Deterministic Policy Gradient (DDPG)
        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.
            actor_critic: The constructor method for a PyTorch Module with an ``act`` 
                method, a ``policy`` module, and a ``q`` module. The ``act`` method and
                ``policy`` module should accept batches of observations as inputs,
                and ``q`` should accept a batch of observations and a batch of 
                actions as inputs. When called, these should return:
                ===========  ================  ======================================
                Call         Output Shape      Description
                ===========  ================  ======================================
                ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                               | observation.
                ``policy``       (batch, act_dim)  | Tensor containing actions from policy
                                               | given observations.
                ``q``        (batch,)          | Tensor containing the current estimate
                                               | of Q* for the provided observations
                                               | and actions. (Critical: make sure to
                                               | flatten this!)
                ===========  ================  ======================================
            seed (int): Seed for random number generators.
            replay_size (int): Maximum length of replay buffer.
            gamma (float): Discount factor. (Always between 0 and 1.)
            polyak (float): Interpolation factor in polyak averaging for target 
                networks. Target networks are updated towards main networks 
                according to:
                .. math:: \\theta_{\\text{targ}} \\leftarrow 
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta
                where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
                close to 1.)
            lr (float): Learning rate
            batch_size (int): Minibatch size for SGD.
        """

        super().__init__()

        self.env = env
        self.polyak = polyak
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.noise_scale = noise_scale
        self.memory_size = memory_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.actor_update = actor_update

        self.inside_dim = inside_dim
        self.num_hidden_layers = num_hidden_layers

        self.seed = seed

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
            obs_dim=env.observation_dim,
            act_dim=env.action_dim,
            size=self.memory_size,
            batch_size=self.batch_size,
        )

        # Set up optimizers for policy and q-function
        self.policy_optimizer = Adam(self.ac.policy.parameters(), lr=self.lr)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=self.lr)

        # transition to store in memory
        self.transition = list()

    def compute_loss_q(self, data):
        obs, action, reward, next_obs, done = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        q = self.ac.q(obs, action)
        # Bellman backup for Q function
        # with torch.no_grad():
        q_policy_targ = self.ac_targ.q(next_obs, self.ac_targ.policy(next_obs))
        backup = reward + self.gamma * (1 - done) * q_policy_targ

        # MSE loss against Bellman backup
        loss_q = F.mse_loss(q, backup)

        return loss_q

    # Set up function for computing DDPG policy loss
    def compute_loss_policy(self, data):
        obs = data["obs"]
        q_policy = self.ac.q(obs, self.ac.policy(obs))
        return -q_policy.mean()

    # Set up model saving

    def update(self):

        data = self.replay_buffer.sample_batch()
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()
        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in self.ac.q.parameters():
            p.requires_grad = False
        # Next run one gradient descent step for policy.
        self.policy_optimizer.zero_grad()
        loss_policy = self.compute_loss_policy(data)
        loss_policy.backward()
        self.policy_optimizer.step()
        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.ac.q.parameters():
            p.requires_grad = True

        self.update_target()

        return loss_policy.item()

    def update_target(self):
        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def select_action(self, obs: np.ndarray) -> np.ndarray:

        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.ac.act(torch.as_tensor(obs, dtype=torch.float32))

        # TODO, check if we uncommment this => add noise_scale to log_dict

        ##if not (self.is_test):
        ##    selected_action += self.noise_scale * np.random.randn(self.act_dim)

        selected_action = np.clip(selected_action, self.env.min_temp, self.env.max_temp)

        if not (self.is_test):
            self.transition = [obs, selected_action]

        return selected_action

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:

        next_state, reward, done, info = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.replay_buffer.store(*self.transition)

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

        if num_iterations is None:
            num_iterations = self.env.numsteps

        if num_iterations > self.env.numsteps:
            print(
                f"WARNING: Number of iterations chosen ({num_iterations}) is higher than the number of steps of the environment ({self.env.numsteps}) "
            )
            num_iterations = self.env.numsteps

        if num_iterations < self.memory_size:
            print(
                f"WARNING: Number of iterations chosen ({num_iterations}) is smaller than the size of the replay buffer ({self.memory_size}) "
            )

            return (None, None)

        logger = SimpleLogger(
            logging_path=logging_path,
            agent_name="DDPG_Agent",
            num_episodes=num_episodes,
            num_iterations=num_iterations,
        )

        # plotting options (make sure the dictionary is in the same order as the columns of the outputted summary_df)
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

            ## to keep track of number of updates and update target network accordingly
            update_cnt = 0

            for i in range(num_iterations):
                action = self.select_action(state)
                next_state, reward, done, info = self.step(action)

                if i % 1000 == 0:
                    print(f"Iteration{i}")
                ## keeping track of the value we've seen
                rewards.append(reward)
                actions.append(action)
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
                if not (self.is_test) and (len(self.replay_buffer) >= self.batch_size):

                    # linearly decrease epsilon
                    self.epsilon = max(
                        self.min_epsilon,
                        self.epsilon
                        - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay,
                    )
                    epsilons.append(self.epsilon)

                    if update_cnt % self.actor_update == 0:
                        loss = self.update()
                        losses.append(loss)

                    # if hard update is needed
                    if update_cnt % self.target_update == 0:
                        self.update_target()

                    update_cnt += 1

            ## slicing lower and upper bound
            lower = episode_num * num_iterations
            upper = (episode_num + 1) * num_iterations

            len_difference = len(tair) - len(epsilons)
            pad_epsilon = [epsilons[0] for i in range(len_difference)]
            epsilons = pad_epsilon + epsilons

            ## extend the length of the loss array such that it is of correct
            # size for plotting
            temp_losses = [loss for loss in losses for _ in range(self.actor_update)]
            len_difference = len(tair) - len(temp_losses)
            pad_losses = [0 for i in range(len_difference)]
            temp_losses = pad_losses + temp_losses
            summary_df = pd.DataFrame(
                {
                    "Tair": tair[lower:upper],
                    "Tset": actions[lower:upper],
                    "PMV": pmv[lower:upper],
                    "Heating": qheat[lower:upper],
                    "Reward": rewards[lower:upper],
                    "Occ": occ[lower:upper],
                    "Loss": temp_losses[lower:upper],
                    "Epsilon": epsilons[lower:upper],
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

        # plot a summary that contatenates all episodes together for a complete overview of the training

        ## extend the length of the loss array such that it is of correct
        # size for plotting
        temp_losses = [loss for loss in losses for _ in range(self.actor_update)]
        len_difference = len(tair) - len(temp_losses)
        pad_losses = [0 for i in range(len_difference)]
        temp_losses = pad_losses + temp_losses

        summary_df = pd.DataFrame(
            {
                "Tair": tair,
                "Tset": actions,
                "PMV": pmv,
                "Heating": qheat,
                "Reward": rewards,
                "Occ": occ,
                "Loss": temp_losses,
                "Epsilon": epsilons,
            }
        )

        summary_df["Reward"] = summary_df["Reward"].apply(lambda x: float(x[0]))

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
        self, logging_path: str, num_iterations=None, num_episodes=1, log=True
    ) -> Tuple[str, pd.DataFrame]:

        return self.train(
            is_test=True,
            logging_path=logging_path,
            num_iterations=num_iterations,
            num_episodes=num_episodes,
            log=log,
        )

    def seed_agent(self, seed):

        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        np.random.seed(seed)

        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)

        return

    # Making a save method to save a trained model
    def save(self, filename, directory):
        torch.save(self.ac.q.state_dict(), "%s/%s_ddpg_q.pth" % (directory, filename))
        torch.save(
            self.ac.policy.state_dict(), "%s/%s_ddpg_policy.pth" % (directory, filename)
        )
        # target
        torch.save(
            self.ac_targ.q.state_dict(),
            "%s/%s_ddpg_q_target.pth" % (directory, filename),
        )
        torch.save(
            self.ac_targ.policy.state_dict(),
            "%s/%s_ddpg_policy_target.pth" % (directory, filename),
        )

    # Making a load method to load a pre-trained model
    def load(self, filename, directory):
        self.ac.q.load_state_dict(
            torch.load("%s/%s_ddpg_q.pth" % (directory, filename))
        )
        self.ac.policy.load_state_dict(
            torch.load("%s/%s_ddpg_policy.pth" % (directory, filename))
        )

        # target

        self.ac_targ.q.load_state_dict(
            torch.load("%s/%s_ddpg_q_target.pth" % (directory, filename))
        )
        self.ac_targ.policy.load_state_dict(
            torch.load("%s/%s_ddpg_policy_target.pth" % (directory, filename))
        )

    def log_dict(self) -> Dict[str, Any]:

        log_dict = {
            "is_test": self.is_test,
            "memory_size": self.memory_size,
            "batch_size": self.batch_size,
            "target_update": self.target_update,
            "actor_update": self.actor_update,
            "epsilon_decay": self.epsilon_decay,
            "max_epsilon": self.max_epsilon,
            "min_epsilon": self.min_epsilon,
            "lr": self.lr,
            "gamma": self.gamma,
            "polyak": self.polyak,
            "inside_dim": self.inside_dim,
            "num_hidden_layers": self.num_hidden_layers,
            "seed": self.seed,
        }

        return log_dict


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
        policy_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.policy = mlp(policy_sizes, activation, nn.Sigmoid)
        self.min_action = min_action
        self.max_action = max_action

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.min_action + self.policy(obs) * (self.max_action - self.min_action)


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
        inside_dim: int = 256,
        num_hidden_layers: int = 2,
        activation=nn.ReLU,
    ):
        super().__init__()

        hidden_sizes = [inside_dim] * num_hidden_layers

        # build policy and value functions
        self.policy = MLPActor(
            obs_dim, act_dim, hidden_sizes, activation, min_action, max_action
        )
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.policy(obs.T).numpy()


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size, batch_size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size, self.batch_size = 0, 0, size, batch_size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs.ravel()
        self.obs2_buf[self.ptr] = next_obs.ravel()
        self.act_buf[self.ptr] = act.ravel()
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self):
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def __len__(self) -> int:
        return self.size


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
