import torch
from torch import  nn

class MyLinear(nn.Module):
    def __init__(self,in_units,units):
        super(MyLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_units,units))   # if not use nn.Parameter(), then weight will not be learned
        self.bias = nn.Parameter(torch.ones(units,))

    def forward(self,x):
        return torch.relu(x @ self.weight + self.bias)

net = nn.Sequential(nn.LazyLinear(4),nn.ReLU(),MyLinear(4,2))
x = torch.randn(2,4)
print(net(x))
print(net[2].weight)