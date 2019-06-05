import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import gym
import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque
from torchreinforce import DeepReinforceModule

env = gym.make('LunarLander-v2')
env.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPISODES = 300
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

class DQN(nn.Module):
    def __init__(self, state_size, action_size, seed, **kwargs):
        super(DQN, self).__init__(**kwargs)
        self.seed = torch.manual_seed(seed)
        self.net = torch.nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.net(x)


policy = DQN(state_size, action_size, seed=0)
target = DQN(state_size, action_size, seed=0)

agent = DeepReinforceModule(state_size, action_size, seed=0, policy_net=policy, target_net=target)

scores = []                        # list containing scores from each episode
scores_window = deque(maxlen=100)
for i_episode in range(EPISODES):
    total_reward = 0
    state = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        score += reward 
    scores_window.append(score)       # save most recent score
    scores.append(score)
    agent.epsilon = max(agent.epsilon_max, agent.epsilon_decay*agent.epsilon)

    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
    if i_episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
env.close()