# -*- coding: utf-8 -*-
from copy import copy, deepcopy
import os
import pickle
from glob import glob

import numpy as np
from collections import deque
from typing import List, Optional, Tuple
import torch
import torch.optim as optim
from torch.nn import SmoothL1Loss

from models.actor_critic import DeterministicPolicyNetwork, FullyConnectedQNetwork
from experiences import ReplayBuffer, ExperienceBatch


ACTOR_FN_PREFIX = "drlnd_p3_actor"
CRITIC_FN_PREFIX = "drlnd_p3_critic"


class DDPGAgent:
    def __init__(
        self,
        actor_state_size: int,
        actor_action_size: int,
        critic_state_size: int,
        critic_action_size: int,
        actor_hidden_layer_dimensions: Tuple[int] = (256, 128),
        critic_hidden_layer_dimensions: Tuple[int] = (256, 128),
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        seed: Optional[int] = None,
    ):
        """
        Creates an instance of a DDPG agent.

        :param actor_state_size: size of state space for actors.
        :param actor_action_size: size of action space for actors.
        :param actor_state_size: size of state space for critics.
        :param actor_action_size: size of action space for critics.
        :param actor_hidden_layer_dimensions: hidden layer dimensions of the policy network.
        :param critic_hidden_layer_dimensions: hidden layer dimensions of Q-network.
        :param lr_actor: learning rate of the policy network.
        :param lr_critic: learning rate of the Q-network.
        :param seed: random seed.
        """
        self.actor_state_size = actor_state_size
        self.actor_action_size = actor_action_size
        self.critic_state_size = critic_state_size
        self.critic_action_size = critic_action_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.critic_local = FullyConnectedQNetwork(
            input_dim=self.critic_state_size,
            output_dim=self.critic_action_size,
            hidden_dims=critic_hidden_layer_dimensions,
            seed=self.seed,
        ).to(self.device)
        self.critic_target = FullyConnectedQNetwork(
            input_dim=self.critic_state_size,
            output_dim=self.critic_action_size,
            hidden_dims=critic_hidden_layer_dimensions,
            seed=self.seed,
        ).to(self.device)

        self.actor_local = DeterministicPolicyNetwork(
            input_dim=self.actor_state_size,
            output_dim=self.actor_action_size,
            hidden_dims=actor_hidden_layer_dimensions,
            seed=self.seed,
        ).to(self.device)
        self.actor_target = DeterministicPolicyNetwork(
            input_dim=self.actor_state_size,
            output_dim=self.actor_action_size,
            hidden_dims=actor_hidden_layer_dimensions,
            seed=self.seed,
        ).to(self.device)

        self.value_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=self.lr_critic
        )
        self.policy_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=self.lr_actor
        )

        self.loss_fn = SmoothL1Loss()

    def act(self, state: np.ndarray, eps: float = 0.0) -> np.ndarray:
        """
        Returns actions for given state as per current policy.

        :param state: current state.
        :param eps: noise weighting coefficient.
        :return: selected action.
        """

        state = torch.from_numpy(state).float().to(self.device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        action += self._sample_noise() * eps

        return np.clip(action, -1, 1)

    def _sample_noise(self) -> np.ndarray:
        """
        Samples noise from a Gaussian distribution.

        :return: ndarray with Gaussian distributed values of the same size as
        the action space.
        """
        return np.random.randn(self.actor_action_size)


class MADDPG:
    """
    A multi-agent deep deterministic policy-gradient agent.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        n_agents: int,
        actor_hidden_layer_dimensions: Tuple[int] = (128, 64),
        critic_hidden_layer_dimensions: Tuple[int] = (512, 256, 128),
        buffer_size: int = 1000_000,
        batch_size: int = 1024,
        gamma: float = 0.99,
        tau: float = 0.1,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-4,
        update_every: int = 2,
        seed: Optional[int] = None,
    ):
        """
        Creates an instance of a DDPG agent.

        :param state_size: size of state space.
        :param action_size: size of action space.
        :param n_agents: number of DDPG agents.
        :param actor_hidden_layer_dimensions: hidden layer dimensions of the policy network.
        :param critic_hidden_layer_dimensions: hidden layer dimensions of Q-network.
        :param buffer_size: replay buffer size.
        :param batch_size: mini-batch size.
        :param gamma: discount factor.
        :param tau: interpolation parameter for target-network weight update.
        :param lr_actor: learning rate of the policy network.
        :param lr_critic: learning rate of the Q-network.
        :param update_every: update every n steps.
        :param seed: random seed.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents
        self.actor_hidden_layer_dimensions = actor_hidden_layer_dimensions
        self.critic_hidden_layer_dimensions = critic_hidden_layer_dimensions
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.memory = ReplayBuffer(
            self.action_size, self.buffer_size, self.batch_size, self.seed
        )

        self.update_every = update_every
        self.step_count = 0
        self.agents: List[DDPGAgent]

    def __getstate__(self):
        state = self.__dict__
        del state["agents"]
        del state["device"]
        del state["memory"]
        return state

    def _initialize_agents(self):
        """
        Initializes agents.
        """
        self.agents = [
            DDPGAgent(
                actor_state_size=self.state_size,
                actor_action_size=self.action_size,
                critic_state_size=self.state_size * self.n_agents,
                critic_action_size=self.action_size * self.n_agents,
                actor_hidden_layer_dimensions=self.actor_hidden_layer_dimensions,
                critic_hidden_layer_dimensions=self.critic_hidden_layer_dimensions,
                lr_actor=self.lr_actor,
                lr_critic=self.lr_critic,
                seed=self.seed,
            )
            for _ in range(self.n_agents)
        ]

    def _step(
        self,
        states: np.ndarray,
        actions: List[np.ndarray],
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """
        Adds the experience to memory and fits the agent.

        :param states: states of the environment.
        :param actions: actions taken.
        :param rewards: rewards received.
        :param next_states: next states after taken action.
        :param dones: indicates if the episode has finished.
        """
        self.memory.add(states, np.concatenate(actions), rewards, next_states, dones)
        self.step_count += 1

        if (
            len(self.memory) > self.batch_size
            and (self.step_count % self.update_every) == 0
        ):
            self._optimize()

    def _agent_states(self, agent_id: int, states: np.ndarray) -> np.ndarray:
        """
        Returns a single agents observation from a global observation.

        :param agent_id: the agent's id.
        :param states: global observations.
        :return: a single agent's observation.
        """
        state_idx_start = self.state_size * agent_id
        state_idx_end = state_idx_start + self.state_size
        return states[:, state_idx_start:state_idx_end]

    def _optimize(self) -> None:
        """
        Updates value parameters using given batch of experience tuples.
        """

        for i, agent in enumerate(self.agents):
            states, actions, rewards, next_states, dones = self.memory.sample()

            actor_next_state = self._agent_states(i, next_states)
            next_actions = torch.cat(
                [a.actor_target(actor_next_state) for a in self.agents], 1
            )
            next_q = agent.critic_target(next_states, next_actions).detach()
            target_q = rewards[:, i].view(-1, 1) + self.gamma * next_q * (
                1 - dones[:, i].view(-1, 1)
            )
            local_q = agent.critic_local(states, actions)

            value_loss = agent.loss_fn(local_q, target_q)
            agent.value_optimizer.zero_grad()
            value_loss.backward()
            agent.value_optimizer.step()

            local_actions = []
            for i, a in enumerate(self.agents):
                local_states = self._agent_states(i, states)
                local_actions.append(
                    a.actor_local(local_states)
                    if a == agent
                    else a.actor_local(local_states).detach()
                )
            local_actions = torch.cat(local_actions, 1)
            policy_loss = -agent.critic_local(states, local_actions).mean()

            agent.policy_optimizer.zero_grad()
            policy_loss.backward()
            agent.policy_optimizer.step()

            self._update_target_model(agent.critic_local, agent.critic_target)
            self._update_target_model(agent.actor_local, agent.actor_target)

    def _update_target_model(self, local_model, target_model) -> None:
        """
        Updates model parameters of target network using Polyak Averaging:

            θ_target = τ*θ_local + (1 - τ)*θ_target

        :param local_model: weights will be copied from.
        :param target_model: weights will be copied to.
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_weight_ratio = (1.0 - self.tau) * target_param.data
            local_weight_ratio = self.tau * local_param.data
            target_param.data.copy_(target_weight_ratio + local_weight_ratio)

    def fit(
        self,
        environment,
        n_episodes: int = 5000,
        max_t: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay: float = 0.999,
        scores_window_length: int = 100,
        average_target_score: float = 0.5,
        agent_checkpoint_dir: Optional[str] = None,
    ) -> List[float]:
        """
        Trains the agent on the given environment.

        :param environment: environment instance to interact with.
        :param n_episodes: maximum number of training episodes.
        :param max_t:  maximum number of time steps per episode.
        :param eps_start: starting value of epsilon, controlling random noise in action selection.
        :param eps_end: minimum value of epsilon.
        :param eps_decay: multiplicative factor (per episode) for decreasing epsilon.
        :param scores_window_length: length of scores window to monitor convergence.
        :param average_target_score: average target score for scores_window_length at which learning stops.
        :param agent_checkpoint_dir: optional directory to store agent's model weights to.
        :return: list of scores.
        """
        self._initialize_agents()
        scores = []
        scores_window = deque(maxlen=scores_window_length)
        eps = eps_start
        for i_episode in range(1, n_episodes + 1):
            states = environment.reset(train_mode=True)
            episode_scores = np.zeros(self.n_agents)
            for t in range(max_t):
                actions = self.act(states, eps)
                next_states, rewards, dones = environment.step(actions)
                self._step(states, actions, rewards, next_states, dones)
                states = next_states
                episode_scores += rewards
                if any(dones):
                    break
            episode_max_score = max(episode_scores)
            scores_window.append(episode_max_score)
            scores.append(episode_max_score)
            eps = max(eps_end, eps_decay * eps)
            average_score_window = float(np.mean(scores_window))
            self._log_progress(
                i_episode, average_score_window, scores_window_length, eps
            )
            if np.mean(scores_window) >= average_target_score:
                print(
                    f"\nEnvironment solved in {i_episode:d} episodes!\t"
                    f"Average Score: {average_score_window:.2f}"
                )
                if agent_checkpoint_dir is not None:
                    self.save(agent_checkpoint_dir)
                break
        return scores

    def act(self, states: np.ndarray, eps: float = 0.0) -> List[np.ndarray]:
        """
        Computes actions for each agent based on given states of the environment.

        :param states: current state.
        :param eps: noise weighting coefficient.
        :return: list of actions from all agents.
        """
        actions = [
            agent.act(state.reshape(-1, 1).T, eps)
            for agent, state in zip(self.agents, states)
        ]
        return actions

    @staticmethod
    def _log_progress(
        i_episode: int,
        average_score_window: float,
        scores_window_length: int,
        eps: float,
    ) -> None:
        """
        Logs average score of episode to stdout.

        :param i_episode: number of current episode.
        :param average_score_window: average score of current episode.
        :param scores_window_length: length of window for computing the average.
        :param eps: current epsilon.
        """
        print(
            f"\rEpisode {i_episode}\tAverage Score: {average_score_window:.2f}, Epsilon: {eps:.2f}",
            end="\n" if i_episode % scores_window_length == 0 else "",
        )

    def save(self, agent_checkpoint_dir: str) -> None:
        """
        Stores the weights of the agents and pickles the MADDPG instance.

        :param agent_checkpoint_dir: path to store agent's model weights to.
        """
        for i, agent in enumerate(self.agents):
            actor_checkpoint_path = os.path.join(
                agent_checkpoint_dir, f"{ACTOR_FN_PREFIX}_{i}.pth"
            )
            critic_checkpoint_path = os.path.join(
                agent_checkpoint_dir, f"{CRITIC_FN_PREFIX}_{i}.pth"
            )
            torch.save(agent.actor_local.state_dict(), actor_checkpoint_path)
            torch.save(agent.critic_local.state_dict(), critic_checkpoint_path)

        with open(os.path.join(agent_checkpoint_dir, f"maddpg.pkl"), "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(agent_checkpoint_dir: str) -> "MADDPG":
        """
        Loads the the pickled MADDPG agent instance.

        :param agent_checkpoint_dir: directory to load the actor model weights from.
        :return: a pre-trained agent instance.
        """
        with open(os.path.join(agent_checkpoint_dir, f"maddpg.pkl"), "rb") as f:
            maddpg = pickle.load(f)
            maddpg._initialize_agents()
            for i, agent in enumerate(maddpg.agents):
                actor_checkpoint_path = os.path.join(
                    agent_checkpoint_dir, f"{ACTOR_FN_PREFIX}_{i}.pth"
                )
                actor_local_state = torch.load(actor_checkpoint_path)
                agent.actor_local.load_state_dict(actor_local_state)

                critic_checkpoint_path = os.path.join(
                    agent_checkpoint_dir, f"{CRITIC_FN_PREFIX}_{i}.pth"
                )
                critic_local_state = torch.load(critic_checkpoint_path)
                agent.critic_local.load_state_dict(critic_local_state)

        return maddpg
