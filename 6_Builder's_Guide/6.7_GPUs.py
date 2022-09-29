import torch
from torch import nn

x = torch.ones(2,3).cuda(torch.device('cuda'))

net = nn.Sequential(nn.LazyLinear(1))
net.to(device=torch.device('cuda'))
print(net(x))

print(net[0].weight.data.device)