import torch
from d2l import torch as d2l


# 1.Relu
x = torch.arange(-8,8,0.1,requires_grad=True)
y = torch.relu(x)
# d2l.plot(x,y,'x','relu(x)',figsize=(5,2.5)) # Error
d2l.plot(x.detach(),y.detach(),'x','relu(x)',figsize=(5,2.5))
d2l.plt.show()
y.backward(torch.ones_like(x),retain_graph=True)    # y.backward(w), compute the gradient of dot(y,w) to x,
# perform the function as "y.sum().backward(retain_grad=True)"
d2l.plot(x.detach(),x.grad,'x','grad of relu',figsize=(5,2.5))
d2l.plt.show()

# 2.Sigmoid
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
d2l.plt.show()
# Clear out previous gradients
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
d2l.plt.show()
# 3.Tanh
y = torch.tanh(x)
d2l.plot(x.detach(),y.detach(),'x','tanh(x)',figsize=(5,2.5))
d2l.plt.show()

x.grad.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(),x.grad,'x','grad of tanh', figsize=(5,2.5))
d2l.plt.show()