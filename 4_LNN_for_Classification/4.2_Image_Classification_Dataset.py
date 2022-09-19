import time
import torch
import torchvision
from torchvision import transforms
from d2l import torch as d2l

# d2l.use_svg_display()
class FashionMNIST(d2l.DataModule):
    def __init__(self,batch_size=64,resize=(28,28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(root=self.root,train=True,transform=trans,download=True)
        self.val = torchvision.datasets.FashionMNIST(root=self.root,train=False,transform=trans,download=True)

data = FashionMNIST(resize=(32,32))
print(f'len of train: {len(data.train)}, len of val: {len(data.val)}')

print(f'shape of each data feature: {data.train[0][0].shape}')
print(f'shape of each data label: {data.train[0][1]}')

@d2l.add_to_class(FashionMNIST)
def text_labels(self,indices):
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[int(i)] for i in indices] # 非常漂亮，统一了标量和数组

@d2l.add_to_class(FashionMNIST)
def get_dataloader(self,train):
    data = self.train if train else self.val
    return torch.utils.data.DataLoader(data,self.batch_size,shuffle=train,num_workers=self.num_workers)

X,y = next(iter(data.train_dataloader()))
print(X.shape,X.dtype,y.shape,y.dtype)

tic = time.time()
for X,y in data.train_dataloader():
    continue
print(f'{time.time()-tic:.3f} sec')

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    raise NotImplementedError

@d2l.add_to_class(FashionMNIST)
def visualize(self, batch, nrows=1, ncols=8, labels=[]):
    X,y = batch
    if not labels:
        labels = self.text_labels(y)
    d2l.show_images(X.squeeze(1),nrows,ncols,titles=labels) # squeeze remove the dimension which size is 1

batch = next(iter(data.val_dataloader()))
data.visualize(batch,nrows=9) # 用nrows和ncols来控制
d2l.plt.show()