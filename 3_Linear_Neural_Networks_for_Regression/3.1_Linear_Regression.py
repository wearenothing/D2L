import math
import time
import numpy as np
import torch
from d2l import torch as d2l

# 1. Vectorization
n = 10000
a = torch.ones(n)
b = torch.ones(n)
c = torch.zeros(n)

t = time.time()
for i in range(n):
    c[i] = a[i] + b[i]
print(f'{time.time() - t:.8f} sec')

t = time.time()
c = a + b
print(f'{time.time() - t:.8f} sec')


# 2. Normal distribution
def normal(x, mu, sigma):
    p = 1 / np.sqrt(2 * np.pi * sigma ** 2)
    return p * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


x = np.arange(-7, 7, 0.01)
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x', ylabel='f(x)',
         legend=[f'mean {mu},std {sigma}' for mu, sigma in params],
         figsize=[8, 6])  # 非常漂亮的写法！！
d2l.plt.show()
