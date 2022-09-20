import torch
from d2l import torch as d2l
from torch import nn
from torch.nn import functional as F

class SoftmaxRegression(d2l.Classifier):
    def __init__(self,num_outputs,lr):
        super(SoftmaxRegression, self).__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),nn.LazyLinear(num_outputs))

    def forward(self,X):
        return self.net(X)

@d2l.add_to_class(d2l.Classifier)
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = Y_hat.reshape(-1,Y_hat.shape[-1])
    Y = Y.reshape(-1)
    return F.cross_entropy(Y_hat,Y,reduction='mean' if averaged else 'none')

data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegression(num_outputs=10, lr=0.01)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
# d2l.plt.ylim(0,0.9)
d2l.plt.show()