import random
from collections import deque
from typing import Tuple, Optional, List
from dataclasses import dataclass

import torch
import numpy as np


ExperienceBatch = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


@dataclass
class Experience:
    """
    Experience data of an agent.
    """

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    """

    def __init__(
        self,
        action_size: int,
        buffer_size: int,
        batch_size: int,
        seed: Optional[int] = None,
    ):
        """
        Creates a ReplayBuffer instance.

        :param action_size: dimension of each action.
        :param buffer_size: maximum size of buffer.
        :param batch_size: size of each training batch.
        :param seed: random seed.
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        if seed is not None:
            random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """
        Adds new experience to the internal memory.

        :param states: current states of the environment.
        :param actions: actions taken.
        :param rewards: rewards received for given action.
        :param next_states: next states after taken the given action.
        :param dones: indicates if episode has finished.
        """
        self.memory.append(Experience(states, actions, rewards, next_states, dones))

    def sample(
        self,
    ) -> ExperienceBatch:
        """
        Randomly sample a batch of experiences from memory.

        :return: batch of experiences.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(
                np.vstack([e.states.flatten() for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.actions.flatten() for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.rewards.flatten() for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack(
                    [e.next_states.flatten() for e in experiences if e is not None]
                )
            )
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(
                np.vstack(
                    [e.dones.flatten() for e in experiences if e is not None]
                ).astype(np.uint8)
            )
            .float()
            .to(self.device)
        )

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """
        Return the current size of internal memory.

        :return: size of internal memory.
        """
        return len(self.memory)
