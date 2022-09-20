import torch
from d2l import torch as d2l


# 1.The Softmax
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1,keepdim=True)
    return X_exp/partition  # The broadcasting mechanism is applied here
# 2.Model
class SoftmaxRegressionScratch(d2l.Module):
    def __init__(self,num_inputs,num_outputs,lr,sigma=0.01):
        super(SoftmaxRegressionScratch, self).__init__()
        self.save_hyperparameters()
        self.w = torch.normal(mean=0,std=sigma,size=(num_inputs,num_outputs),requires_grad=True)
        self.b = torch.zeros(num_outputs,requires_grad=True)

    def parameters(self):
        return [self.w,self.b]

    def forward(self, X):
        return softmax(X.reshape(-1,self.w.shape[0]) @ self.w + self.b)  # Every row share the same w and b, so b is broadcasting along axis1


# 3.The Cross-Entropy Loss
def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y]).mean() # Distinguish from y_hat[:,y], which get the first and third eles of every row.

@d2l.add_to_class(SoftmaxRegressionScratch)
def loss(self,y_hat,y):
    return cross_entropy(y_hat,y)

y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print(cross_entropy(y_hat,y))
# 4.Training

data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegressionScratch(num_inputs=784,num_outputs=10,lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model,data) #TODO: val acc is not drawn


# 5.Prediction

X,y = next(iter(data.val_dataloader()))
preds = model(X).argmax(axis=1)
wrong = (preds.type(y.dtype) != y)  # Use bool matrix to index, the corresponding position with value True will be reserved
X, y, preds = X[wrong], y[wrong], preds[wrong]
labels = [a + '\n' + b for a,b in zip(data.text_labels(y),data.text_labels(preds))]
data.visualize([X,y], labels=labels)
d2l.plt.show()
