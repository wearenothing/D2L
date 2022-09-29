import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.LazyLinear(8)
        self.out = nn.LazyLinear(2)

    def forward(self,x):
        return self.out(F.relu(self.hidden(x)))

x = torch.randn(2,4)
net = MLP()
y = net(x)

torch.save(net.state_dict(),'mlp.params')

clone = MLP()
clone.load_state_dict(torch.load('mlp.params')) # load first
print(clone(x) == y)
