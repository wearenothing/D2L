import torch

# 1.Scalar,Vector,Matrix,Tensor
x1 = torch.tensor(3)
x2 = torch.arange(3)  # tensor([0,1,2])
x3 = torch.arange(6).reshape(2, 3)
x4 = torch.arange(24).reshape(2, 3, 4)

# 2.Sum,Mean,Cumsum
a1 = x3.sum(axis=0)
a2 = x3.sum(axis=1)
a3 = x3.sum(axis=0, keepdim=True)
a4 = x3.sum(axis=1, keepdim=True)
a5 = x3.sum()
# 3.Dot,A×B,A@B
x = torch.arange(3, dtype=torch.float32)
y = torch.ones(3, dtype=torch.float32)
print(torch.dot(x, y))

print(x*y)

x = x.reshape(-1,1)
y = y.reshape(-1,1)
print(x@y.T)

# 4.Norm
# x.norm() = torch.sqrt(torch.dot(x,x))
u = torch.tensor([-3., 4])
print(torch.norm(u))