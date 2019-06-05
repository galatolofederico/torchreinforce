import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
import random

from .replay_memory import *


class DeepReinforceModule(nn.Module):
    def __init__(self, state_size, action_size, policy_net, target_net, **kwargs):
        super(DeepReinforceModule, self).__init__()
        #Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #Iperparametri
        self.gamma = kwargs.get("gamma", 0.99)
        self.tau = kwargs.get("tau", 1e-3)
        self.lr = kwargs.get("learning_rate", 5e-4)
        self.epsilon = kwargs.get("epsilon", 1.0)
        self.epsilon_max = kwargs.get("epsilon_max", 0.01)
        self.epsilon_decay = kwargs.get("epsilon_decay", 0.995)
        self.update_rate = kwargs.get("update_rate", 4)
        self.memory_size = kwargs.get("memory_size", int(1e5))
        self.batch_size = kwargs.get("batch_size", 64)

        #Parametri reti
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(kwargs.get("seed", 0))

        #Reti
        self.qnetwork_policy = policy_net
        self.qnetwork_target = target_net
        self.optimizer = optim.Adam(self.qnetwork_policy.parameters(), lr=self.lr)

        self.memory = ReplayMemory(self.memory_size, self.batch_size)
        self.counter = 0
            
        

    def step(self, state, action, reward, next_state, done):
        '''Take a sampled batch from replay memory and compute Q(s, a) and V(s). Return a MSE loss between Q and V.'''
        self.memory.store(state, action, reward, next_state, done)
        self.counter = (self.counter + 1) % self.update_rate
        if self.counter == 0:
            if len(self.memory) > self.batch_size:
                self.optimize()


    def select_action(self, state):
        '''Perform an annealing epsilon-greedy policy'''
        state = torch.as_tensor(state, dtype=torch.float).unsqueeze(0)
        self.qnetwork_policy.eval()
        with torch.no_grad():
            action_values = self.qnetwork_policy(state)
        self.qnetwork_policy.train()
        if random.random() > self.epsilon:
            _, max_action = torch.max(action_values, 1)
            return max_action.item()
            #np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(range(self.action_size))

    def optimize(self):
        experiences = self.memory.sample(self.device)

        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_policy(states).gather(1, actions)
        loss = functional.mse_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update()
    
    def soft_update(self):
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_policy.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
    
    def epsilon_annealign(self):
        self.epsilon = max(self.epsilon_max, self.epsilon*self.epsilon_decay)
        #self.epsilon = self.epsilon_max + (self.epsilon - self.epsilon_max) * math.exp(-1 * self.counter / self.epsilon_decay)