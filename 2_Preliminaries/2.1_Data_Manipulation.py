import torch
# Create tensor
x1 = torch.zeros(3, 4)
x2 = torch.arange(12, dtype=torch.float32)

# Concatenate
torch.exp(x1)
torch.cat((x1,x2.reshape(3,4)),dim=0)
torch.cat((x1,x2.reshape(3,4)),dim=1)

# Convert
x = x1.numpy()
x3 = torch.from_numpy(x)





