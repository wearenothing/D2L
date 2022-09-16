import torch
from torch import nn
from d2l import torch as d2l


class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, num_inputs)
        noise = torch.randn(n, 1) * 0.01
        w, b = torch.ones(num_inputs, 1) * 0.01, 0.05
        self.y = self.X @ w + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)


def l2_penalty(w):
    return (w ** 2).sum() / 2

class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self,num_inputs,lambd, lr, sigma=0.01):
        super(WeightDecayScratch, self).__init__(num_inputs,lr,sigma)
        self.save_hyperparameters()

    def loss(self,y_hat,y):
        return super(WeightDecayScratch, self).loss(y_hat,y) + self.lambd * l2_penalty(self.w)

data = Data(20,100,200,5)
trainer = d2l.Trainer(max_epochs=10)
def train_scratch(lambd):
    model  = WeightDecayScratch(200,lambd,0.01)
    model.board.yscale = 'log'
    trainer.fit(model,data)
    print('L2 norm of w:',float(l2_penalty(model.w)))

# train_scratch(0)

# train_scratch(3)

class WeightDecay(d2l.LinearRegression):
    def __init__(self,wd,lr):
        super(WeightDecay, self).__init__(lr)
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.SGD(self.net.parameters(),lr=self.lr,weight_decay=self.wd)


model = WeightDecay(wd=3, lr=0.01)
model.board.yscale='log'
trainer.fit(model, data)
print('L2 norm of w:', float(l2_penalty(model.get_w_b()[0])))