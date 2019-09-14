import torch
from .base import ReinforceDistribution

class Categorical(ReinforceDistribution, torch.distributions.Categorical):
    def __init__(self, probs, **kwargs):
        self.deterministic = kwargs.pop("deterministic", False)
        self.probs = probs
        torch.distributions.Categorical.__init__(self, probs, **kwargs)

    def sample(self):
        if self.deterministic:
            return self.probs.max(len(self.probs.shape) - 1)[1]
        else:
            return torch.distributions.Categorical.sample(self)
