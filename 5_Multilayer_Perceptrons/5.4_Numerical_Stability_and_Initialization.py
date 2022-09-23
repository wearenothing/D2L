import torch
from d2l import torch as d2l

# Vanishing Gradient
x = torch.arange(-8,8,0.1,requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach(),[y.detach(),x.grad],legend=['sigmoid','gradient'],figsize=(6,4.5))
d2l.plt.show()

# Exploding Gradient
M = torch.normal(0,1,size=(4,4))
print('Before 100 times multiplying',M)
for i in range(100):
    M = M @ torch.normal(0,1,size=(4,4))

print('After 100 times multiplying',M)
