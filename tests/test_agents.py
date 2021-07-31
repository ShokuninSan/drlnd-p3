# -*- coding: utf-8 -*-
import os

import pytest
import numpy as np
from unityagents import UnityEnvironment

from environments import UnityEnvWrapper
from agents.policy_based import MADDPG


ENVIRONMENT_PATH = os.path.join("..", "environments", "Tennis.app")
TEST_ENV = UnityEnvWrapper(UnityEnvironment(file_name=ENVIRONMENT_PATH))


def test_random_agent():

    num_agents = 2
    action_size = 2

    for i in range(1, 6):  # play game for 5 episodes
        states = TEST_ENV.reset(train_mode=False)  # reset the environment
        scores = np.zeros(num_agents)  # initialize the score (for each agent)
        while True:
            actions = np.random.randn(
                num_agents, action_size
            )  # select an action (for each agent)
            actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
            next_states, rewards, dones = TEST_ENV.step(
                actions
            )  # send all actions to tne environment
            scores += rewards  # update the score (for each agent)
            states = next_states  # roll over states to next time step
            if np.any(dones):  # exit loop if episode finished
                break
        print("Score (max over agents) from episode {}: {}".format(i, np.max(scores)))


def test_maddpg_agent():
    num_agents = 2

    agent = MADDPG(TEST_ENV.state_size, TEST_ENV.action_size, num_agents)
    agent.fit(
        environment=TEST_ENV,
        average_target_score=0.5,
    )
