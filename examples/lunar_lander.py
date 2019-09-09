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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('LunarLander-v2')
env.seed(0)

EPISODES = 1000
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

class DQN(nn.Module):
    def __init__(self, state_size, action_size, **kwargs):
        super(DQN, self).__init__(**kwargs)
        self.seed = torch.manual_seed(kwargs.get("seed", 0))
        self.net = torch.nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.net(x)
    
    def to(self, device):
        self.net = self.net.to(device)
        return self


policy = DQN(state_size, action_size).to(device)
target = DQN(state_size, action_size).to(device)

agent = DeepReinforceModule(policy_net=policy, target_net=target, device=device)

scores_window = deque(maxlen=100)
for i_episode in range(EPISODES):
    state = env.reset()
    done = False
    score = 0
    i = 0
    while not done:
        action = agent.select_action(state, action_size)
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
    scores_window.append(score)
    agent.epsilon_annealing()
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
    if i_episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
env.close()

env.reset()
while True:
    action = agent.select_action(state, action_size)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    env.render()
    if done:
        state = env.reset()
env.close()