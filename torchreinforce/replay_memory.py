import torch
from collections import namedtuple
import random

class ReplayMemory:
    def __init__(self, memory_size, batch_size, seed=0):
        self.memory = []
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.transition = namedtuple('Transition', 'state action reward next_state done')

    
    def store(self, state, action, reward, next_state, done):
        transition = self.transition(state, action, reward, next_state, done)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        self.memory.append(transition)
    
    def sample(self, device):
        transitions = random.sample(self.memory, k=self.batch_size)

        states = torch.tensor([t.state for t in transitions if t is not None], dtype=torch.float, device=device).unsqueeze(1).squeeze(1)
        actions = torch.tensor([t.action for t in transitions if t is not None], dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in transitions if t is not None], dtype=torch.float, device=device).unsqueeze(1)
        next_states = torch.tensor([t.next_state for t in transitions if t is not None], dtype=torch.float, device=device).unsqueeze(1).squeeze(1)
        dones = torch.tensor([t.done for t in transitions if t is not None], dtype=torch.float, device=device).unsqueeze(1)
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)