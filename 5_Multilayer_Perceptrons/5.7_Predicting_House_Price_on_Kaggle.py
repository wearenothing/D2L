import torch
from torch import nn
from d2l import torch as d2l
import pandas as pd
import numpy as np


# 1.Accessing and Reading the Dataset
class KaggleHouse(d2l.DataModule):
    def __init__(self, batch_size, train=None, val=None):
        super(KaggleHouse, self).__init__()
        self.save_hyperparameters()
        if train is None:
            self.raw_train = pd.read_csv(d2l.download(d2l.DATA_URL + 'kaggle_house_pred_train.csv',
                                                      sha1_hash='585e9cc93e70b39160e7921475f9bcd7d31219ce'))
            self.raw_val = pd.read_csv(d2l.download(d2l.DATA_URL + 'kaggle_house_pred_test.csv',
                                                    sha1_hash='fa19780a7b011d9b009e8bff8e99922a8ee2eb90'))


data = KaggleHouse(batch_size=64)
print(data.raw_train.shape)
print(data.raw_val.shape)

print(data.raw_train.iloc[:4, [0, 1, 2, 3, -3, -2, -1]])


@d2l.add_to_class(KaggleHouse)
def preprocess(self):
    # Remove the ID and label columns
    label = 'SalePrice'
    features = pd.concat(
        [self.raw_train.drop(columns=['Id', label]), self.raw_val.drop(columns=['Id'])])  # Need an extra [] or ()
    numeric_features = features.dtypes[features.dtypes != 'object'].index
    # Standardize numerical columns
    features[numeric_features] = features[numeric_features].apply(
        lambda x: (x - x.mean()) / x.std())  # Along the columns on default
    # Replace NAN numerical features by 0
    features[numeric_features] = features[numeric_features].fillna(0)  # fillna
    # Replace discrete features by one-hot encoding.
    features = pd.get_dummies(features, dummy_na=True)  # get_dummies only works on non-numerical features
    # Save preprocessed features
    self.train = features[:self.raw_train.shape[
        0]].copy()  # copy: Make sure the change of self.train has no effect on original data
    self.train[label] = self.raw_train[label]
    self.val = features[self.raw_train.shape[0]:].copy()


data.preprocess()
print(data.train.shape)


# print(data.label.shape)
# print(data.val.shape)

@d2l.add_to_class(KaggleHouse)
def get_dataloader(self, train):
    label = 'SalePrice'
    data = self.train if train else self.val
    if label not in data:
        return
    get_tensor = lambda x: torch.tensor(x.values, dtype=torch.float32)  # Convert from DataFrame to Tensor
    # Logarithm of prices
    tensors = (get_tensor(data.drop(columns=[label])),  # X
               torch.log(get_tensor(data[label])).reshape(-1, 1))  # y
    return self.get_tensorloader(tensors, train)


X, y = next(iter(data.get_dataloader(train=True)))
print(f'shape of X:{X.shape},shape of y:{y.shape}')


# 2.K-Fold Cross-Validation
def k_fold_data(data, k):
    # rets
    rets = []
    # fold_size
    fold_size = data.train.shape[0] // k
    # idx
    for j in range(k):
        idx = range(j * fold_size, (j + 1) * fold_size)
        rets.append(KaggleHouse(data.batch_size, data.train.drop(index=idx),
                                data.train.loc[idx]))  # data.train.drop as self.train; data.train.loc[idx] as validation (not val, which is actually test)
    return rets


def k_fold(trainer, data, k, lr):
    val_loss, models = [], []
    for i, data_fold in enumerate(k_fold_data(data, k)):
        model = d2l.LinearRegression(lr)
        model.board.yscale = 'log'
        if i != 0: model.board.display = False  # Only draw the first graph
        trainer.fit(model, data_fold)
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        models.append(model)
    print(f'average validation log mse = {sum(val_loss) / len(val_loss)}')
    return models


# 3.Training

trainer = d2l.Trainer(max_epochs=10)
models = k_fold(trainer, data, 5, 0.01)
# trainer.fit(models, data)

preds = [model(torch.tensor(data.val.values,dtype=torch.float32)) for model in models]
ensemble_preds = torch.exp(torch.cat(preds,dim=1)).mean(dim=1)
submission = pd.DataFrame({'Id':data.raw_val.Id, 'SalePrice':ensemble_preds.detach().numpy()})
submission.to_csv('..\\data\\submissions\\house_price_submission.csv',index=False)

d2l.plt.show()
