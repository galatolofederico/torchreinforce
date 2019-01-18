import torch

class ReinforceDistribution:
    def __init__(self):
        pass

class Categorical(ReinforceDistribution, torch.distributions.Categorical):
    def __init__(self, probs, **kwargs):
        self.deterministic = kwargs["deterministic"] if "deterministic" in kwargs else False
        self.probs = probs
        if "deterministic" in kwargs: del kwargs["deterministic"]
        torch.distributions.Categorical.__init__(self, probs, **kwargs)
        
    def sample(self):
        if self.deterministic:
            return self.probs.max(0)[1]
        else:
            return torch.distributions.Categorical.sample(self)

def getNonDeterministicWrapper(baseClass):
    class NonDeterministicWrapper(baseClass):
        def __init__(self, *args, **kwargs):
            if "deterministic" in kwargs: del kwargs["deterministic"]
            super(NonDeterministicWrapper, self).__init__(*args, **kwargs)
    return NonDeterministicWrapper