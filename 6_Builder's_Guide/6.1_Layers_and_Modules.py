import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


# net = nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
net = nn.Sequential(nn.LazyLinear(256),nn.ReLU(),nn.LazyLinear(10))
X = torch.randn(2,20)
print(net(X))

# 1.A custom module
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.LazyLinear(256)
        self.out = nn.LazyLinear(10)


    def forward(self,X):
        return self.out(torch.relu(self.hidden(X)))

net = MLP()
print(net(X).shape)


# 2.The Sequential Module
class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx),module)    # Adds a child module to the current module.

    def forward(self, X):
        for module in self.children():  # Returns an iterator over immediate children modules.
            X = module(X)
        return X

net = MySequential(nn.LazyLinear(256),nn.ReLU(),nn.LazyLinear(10))
print(net(X).shape)

# 3. Forward
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super(FixedHiddenMLP, self).__init__()
        self.rand_weight = torch.rand(20,20)
        self.linear = nn.LazyLinear(20)

    def forward(self,X):
        X = self.linear(X)
        X = F.relu(X @ self.rand_weight + 1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

net = FixedHiddenMLP()
print(net(X))

class NestMLP(nn.Module):
    def __init__(self):
        super(NestMLP, self).__init__()
        self.net = nn.Sequential(nn.LazyLinear(64),nn.ReLU(),nn.LazyLinear(32),nn.ReLU())
        self.linear = nn.LazyLinear(16)

    def forward(self,X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(),nn.LazyLinear(20),FixedHiddenMLP())
print(chimera(X))

