import torch
from d2l import torch as d2l

x = torch.arange(-8,8,0.1,requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach(),[y.detach(),x.grad],legend=['sigmoid','gradient'],figsize=(6,4.5))
d2l.plt.show()