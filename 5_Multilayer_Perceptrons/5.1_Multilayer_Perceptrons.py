import torch
from matplotlib import pyplot as plt


# 1.Relu
x = torch.arange(-8,8,0.1,requires_grad=True)
y = torch.relu(x)
plt.figure(figsize=(9,6))
plt.subplot(231)
# d2l.plot(x,y,'x','relu(x)',figsize=(5,2.5)) # Error
plt.plot(x.detach(),y.detach())


# Gradient of Relu
y.backward(torch.ones_like(x),retain_graph=True)    # y.backward(w), compute the gradient of dot(y,w) to x,
# perform the function as "y.sum().backward(retain_grad=True)"
plt.subplot(234)
plt.plot(x.detach(),x.grad)


# 2.Sigmoid
y = torch.sigmoid(x)
plt.subplot(232)
plt.plot(x.detach(),y.detach())

# Gradient of sigmoid
x.grad.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
plt.subplot(235)
plt.plot(x.detach(),x.grad)


# 3.Tanh
y = torch.tanh(x)

plt.subplot(233)
plt.plot(x.detach(),y.detach())

x.grad.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
plt.subplot(236)
plt.plot(x.detach(),x.grad)
plt.show()