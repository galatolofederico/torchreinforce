import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
from torchreinforce import ReinforceModule
import gym

EPISODES = 1000

class Model(ReinforceModule):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
            torch.nn.Softmax(dim=-1),
        )
    
    @ReinforceModule.forward
    def forward(self, x):
        return self.net(x)


env = gym.make('CartPole-v0')
net = Model()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

for i in range(EPISODES):
    done = False
    net.reset()
    observation = env.reset()
    while not done:
        action = net(torch.tensor(observation, dtype=torch.float32))
        
        observation, reward, done, info = env.step(action.get())
        action.reward(reward)
        
    loss = net.loss(normalize=False)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_reward = net.total_reward()
    if i%50 == 0:
        print("Episode: %d  loss: %f  total_reward: %d  reward_threshold: %d" % (i, loss.item(), total_reward, env.spec.reward_threshold))


observation = env.reset()
net.reset()
i = 0
while True:
    action = net(torch.tensor(observation, dtype=torch.float32))
    observation, _, done, _ = env.step(action.get())
    env.render()
    i += 1
    if done:
        observation = env.reset()
        print("Episode length: %d" % i)
        i = 0
