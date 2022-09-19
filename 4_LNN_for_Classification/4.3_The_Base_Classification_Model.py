import torch
from d2l import torch as d2l

class Classifier(d2l.Module):
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1]) # 啥意思
        self.plot('loss',self.loss(Y_hat,batch[-1]),train=False)
        self.plot('acc',self.accuracy(Y_hat,batch[-1]),train=False)

@d2l.add_to_class(d2l.Module)
def configure_optimizer(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)

@d2l.add_to_class(Classifier)
def accuracy(self, Y_hat, Y, averaged=True):
    Y_hat = Y_hat.reshape(-1,Y_hat.shape[-1])
    preds = Y_hat.argmax(axis=1).type(Y.dtype)
    compare = (preds == Y.reshape(-1)).type(torch.float32)
    return compare.mean() if averaged else compare


