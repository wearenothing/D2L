import torch
# 1.gard
x = torch.arange(4.0,requires_grad=True)
y = 2 * torch.dot(x,x) # y should be a scalar, so dot is used
y.backward()
print(x.grad)

# 2.detach
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
print(x.grad == u)
