import torch
from torch import nn

net = nn.Sequential(nn.LazyLinear(8),nn.ReLU(),nn.LazyLinear(1))
x = torch.randn(2,4)
print(net(x))

print(net[2].state_dict())
print(net[2].weight.data)
print(net[2].bias.data)
print(net[2].bias)

# Accessing all parameters at once
print([(name,parameter.shape) for name,parameter in net.named_parameters()])

# Tied Parameters
# We need to give the shared layer a name so that we can refer to its parameters
shared = nn.LazyLinear(8)
net = nn.Sequential(nn.LazyLinear(8),nn.ReLU(),shared,nn.ReLU(),shared,nn.ReLU(),nn.LazyLinear(1))
print(net(x))
print(net[2].bias.data == net[4].bias.data)
net[2].bias.data[0] = 5
print(net[2].bias.data == net[4].bias.data)
print(net[2].bias.data)