import torch
from torch import nn
from d2l import torch as d2l
import pandas as pd
import numpy as np

# 1.Accessing and Reading the Dataset
class KaggleHouse(d2l.Classifier):
    def __init__(self,batch_size,train=None,val=None):
        super(KaggleHouse, self).__init__()
        self.save_hyperparameters()
        if train is None:
            self.raw_train = pd.read_csv(d2l.download(d2l.DATA_URL+'kaggle_house_pred_train.csv',sha1_hash='585e9cc93e70b39160e7921475f9bcd7d31219ce'))
            self.raw_val = pd.read_csv(d2l.download(d2l.DATA_URL+'kaggle_house_pred_test.csv',sha1_hash='fa19780a7b011d9b009e8bff8e99922a8ee2eb90'))

data = KaggleHouse(batch_size=64)
print(data.raw_train.shape)
print(data.raw_val.shape)

print(data.raw_train.iloc[:4,[0,1,2,3,-3,-2,-1]])

@d2l.add_to_class(KaggleHouse)
def preprocess(self):
    # Remove the ID and label columns
    label = 'SalePrice'
    features = pd.concat(
        (self.raw_train.drop(columns=['Id', label]),
         self.raw_val.drop(columns=['Id'])))
    # Standardize numerical columns
    numeric_features = features.dtypes[features.dtypes != 'object'].index
    features[numeric_features] = features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    # Replace NAN numerical features by 0
    features[numeric_features] = features[numeric_features].fillna(0)
    # Replace discrete features by one-hot encoding.
    features = pd.get_dummies(features, dummy_na=True)
    # Save preprocessed features
    self.train = features[:self.raw_train.shape[0]].copy()  # copy: Make sure the change of self.train has no effect on original data
    self.train[label] = self.raw_train[label]
    self.val = features[self.raw_train.shape[0]:].copy()
