import torch
from d2l import torch as d2l

X, w_xh = torch.randn(3,1), torch.randn(1,4)
H, w_hh = torch.randn(3,4), torch.randn(4,4)

print(X @ w_xh + H @ w_hh)

print(torch.cat((X,H),1) @ torch.cat((w_xh,w_hh),0))

