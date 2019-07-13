
from pathlib import Path
import requests

import pickle
import gzip

from matplotlib import pyplot
import numpy as np

import torch
import math
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

batch_size = 64
lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

#pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
#pyplot.show()
#print(x_train.shape)

def get_data(features, labels, batch_size):
    data_ds = TensorDataset(features, labels)
    data_dl = DataLoader(data_ds, batch_size=batch_size)
    return data_dl
    return (DataLoader)

x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
#n, c = x_train.shape

#print(x_train, y_train)
#print(x_train.shape)
#print(y_train.min(), y_train.max())

#train_ds = TensorDataset(x_train, y_train)
#train_dl = DataLoader(train_ds, batch_size=batch_size)
#valid_ds = TensorDataset(x_valid, y_valid)
#valid_dl = DataLoader(valid_ds, batch_size=batch_size * 2)
train_dl = get_data(x_train, y_train, batch_size)
valid_dl = get_data(x_valid, y_valid, batch_size * 2)

#weights = torch.randn(784, 10) / math.sqrt(784)
#weights.requires_grad_()

#bias = torch.zeros(10, requires_grad=True)

#def log_softmax(x):
#    return x - x.exp().sum(-1).log().unsqueeze(-1)

#def model(xb):
#    return log_softmax(xb @ weights + bias)

#def model(xb): 
#    return xb @ weights + bias


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        #self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        #self.bias = nn.Parameter(torch.zeros(10))
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        #return xb @ self.weights + self.bias
        return self.lin(xb)

def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr = lr)
model, opt = get_model()


xb = x_train[0 : batch_size]
preds = model(xb)
print(preds[0], preds.shape)


#def negative_log_likelihood(input, target):
#    return -input[range(target.shape[0]), target].mean()

#loss_func = negative_log_likelihood
loss_func = F.cross_entropy

yb = y_train[0 : batch_size]
loss = loss_func(preds, yb)
print(loss)

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

acc = accuracy(preds, yb)
print(acc)

def loss_batch(model, loss_func, xb, yb, opt=None):
    pred = model(xb)
    loss = loss_func(pred, yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


#def fit():
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        #for i in range((n - 1) // batch_size + 1):
            #start_i = i * batch_size
            #end_i = start_i + batch_size
            #xb = x_train[start_i : end_i]
            #yb = y_train[start_i : end_i]
            #xb, yb = train_ds[i * batch_size : i * batch_size + batch_size]
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)
            #pred = model(xb)
            #loss = loss_func(pred, yb)

            #loss.backward()
            #opt.step()
            #opt.zero_grad()
            #with torch.no_grad():
                #weights -= weights.grad * lr
                #bias -= bias.grad * lr
                #weights.grad.zero_()
                #bias.grad.zero_()
                #for p in model.parameters(): 
                #    p -= p.grad * lr
                #model.zero_grad()
        model.eval()
        with torch.no_grad():
            #valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(epoch, val_loss)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

print(loss_func(model(xb), yb), accuracy(model(xb), yb))






