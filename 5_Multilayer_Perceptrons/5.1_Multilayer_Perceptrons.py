import torch
from d2l import torch as d2l

x = torch.arange(-8,8,0.1,requires_grad=True)
y = torch.relu(x)
# d2l.plot(x,y,'x','relu(x)',figsize=(5,2.5)) # Error
d2l.plot(x.detach(),y.detach(),'x','relu(x)',figsize=(5,2.5))
d2l.plt.show()