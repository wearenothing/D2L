import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class RNNScratch(d2l.Module):
    def __init__(self,num_inputs,num_hiddens,sigma=0.01):
        super(RNNScratch, self).__init__()
        self.save_hyperparameters()
        self.W_xh = nn.Parameter(torch.randn(num_inputs,num_hiddens) * sigma)
        self.W_hh = nn.Parameter(torch.randn(num_hiddens,num_hiddens) * sigma)
        self.b_h = nn.Parameter(torch.zeros(num_hiddens))

@d2l.add_to_class(RNNScratch)  #@save
def forward(self, inputs, state=None):
    if state is not None:
        state, = state
    outputs = []
    for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs)
        state = torch.tanh(torch.matmul(X, self.W_xh) + (
            torch.matmul(state, self.W_hh) if state is not None else 0)
                         + self.b_h)
        outputs.append(state)
    return outputs, state

batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = torch.ones((num_steps, batch_size, num_inputs))
outputs, state = rnn(X)
print(outputs, state)

def check_len(a, n):  #@save
    assert len(a) == n, f'list\'s len {len(a)} != expected length {n}'

def check_shape(a, shape):  #@save
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'

d2l.check_len(outputs, num_steps)
d2l.check_shape(outputs[0], (batch_size, num_hiddens))
d2l.check_shape(state, (batch_size, num_hiddens))

