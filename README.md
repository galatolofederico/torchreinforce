# torchreinforce

A pythonic implementation of the REINFORCE algorithm that is actually fun to use

## Installation
You can install it with pip as you would for any other python package
```
git clone https://github.com/galatolofederico/torchreinforce
cd torchreinforce
python setup.py install
```

## Quickstart

In order to use the REINFORCE algorithm with your model you only need to do two things:
* Use the ``ReinforceModule`` class as your base class
* Decorate your ``forward`` function with ``@ReinforceModule.forward``

That's it!

```python
from torchreinforce import ReinforceModule

class Model(ReinforceModule):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(20, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
            torch.nn.Softmax(dim=-1),
        )
    
    @ReinforceModule.forward
    def forward(self, x):
        return self.net(x)
```

Your model will now output ``ReinforceOutput`` objects.

This objects have two important functions

* ``get()``
* ``reward(value)``

You can use ``output.get()`` to get an actual sample of the overlaying distribution and ``output.reward(value)`` to set a reward for the specific output.

Being ``net`` your model you have to do something like that

```python
action = net(observation)
observation, reward, done, info = env.step(action.get())
action.reward(reward)
```

## Wait, did you just said distribution?

Yes! As the REINFORCE algorithm states the outputs of your model will be used as parameters for a probability distribution function.

Actually you can use whatever probability distribution you want, the ``ReinforceModule`` constructor accepts indeed the following parameters:

* ``gamma`` the *gamma* parameter of the REINFORCE algorithm (default: ``Categorical``)
* ``distribution`` every ``ReinforceDistribution`` or ``pytorch.distributions`` distribution (default: 0.99)

like that

```python
net = Model(distribution=torch.distributions.Beta, gamma=0.99)
```

Keep in mind that the outputs of your **decorated** ``forward(x)`` outputs will be used as the parameters for the ``distribution``. If your ``distribution`` needs more than one parameters just return a list.

I've added the possibility to distribution to have a **deterministic** behavior in **testing** and I've implemented it only for the ``Categorical`` distribution, if you want to implement your own deterministic logic check the file ``distributions/categorical.py`` it is pretty straightforward

If you want to use the ``torch.distributions.Beta`` distribution for example you will need to do something like

```python
from torchreinforce import ReinforceModule

class Model(ReinforceModule):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        ...
    
    @ReinforceModule.forward
    def forward(self, x):
        return [self.net1(x), self.net2(x)] # the Beta distribution accepts two parameters

net = Model(distribution=torch.distributions.Beta, gamma=0.99)

action = net(inp)
env.step(action.get())
```

## Nice! What about training?

You can compute the REINFORCE loss by calling the ``loss()`` function of ``ReinforceModule`` and than treat it as you would do with any other pytorch loss function

```python
net = ...
optmizer = ...

while training:
    net.reset()
    for steps:
        ....
    
    loss = net.loss(normalize=True)

    optimizer.zero_grad()
    loss.backward()
    optmizer.step()
```

You **have to** call the ``reset()`` function of ``ReinforceModule`` **before** the beginning of each episode. You can also pass the argument ``normalize`` to ``loss()`` if you want to normalize the rewards

## Putting all together

A complete example looks like this:

```python
from torchreinforce import ReinforceModule

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
```

You can find a running example in the ``examples/`` folder.