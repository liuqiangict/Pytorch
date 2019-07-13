
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

def get_data(features, labels, batch_size):
    data_ds = TensorDataset(features, labels)
    data_dl = DataLoader(data_ds, batch_size=batch_size)
    return data_dl

x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

train_dl = get_data(x_train, y_train, batch_size)
valid_dl = get_data(x_valid, y_valid, batch_size * 2)


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, xb):
        return self.linear(xb)

def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr = lr)

model, opt = get_model()

loss_func = F.cross_entropy


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

def loss_batch(model, loss_func, xb, yb, opt=None):
    pred = model(xb)
    loss = loss_func(pred, yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)
        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print("Epoch {0}, Loss {1}.".format(epoch, val_loss))

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

xb = x_train[0 : batch_size]
yb = y_train[0 : batch_size]
print(loss_func(model(xb), yb), accuracy(model(xb), yb))
