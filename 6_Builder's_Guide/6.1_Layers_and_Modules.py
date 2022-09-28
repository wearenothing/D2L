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