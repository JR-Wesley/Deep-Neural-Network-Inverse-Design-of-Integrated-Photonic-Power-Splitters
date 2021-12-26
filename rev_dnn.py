import torch
import numpy as np
from torch import nn
from torch.utils import data
from torch.nn import functional as func
from d2l import torch as d2l


# init weight
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std = 0.2)


def load_data(data_arrays, size, is_train = True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, size, shuffle = is_train)


class Residual(nn.Module):
    def __init__(self, i_num, h_num, o_num):
        super().__init__()
        self.L1 = nn.Linear(i_num, h_num)
        self.L2 = nn.Linear(h_num, o_num)

    def forward(self, X):
        Y = func.sigmoid(self.L1(X))
        Y = self.L2(Y)
        Y += X
        return Y


size = 50000
labels = torch.tensor(np.loadtxt("B0.txt", max_rows = size * 5).reshape(size, 25)).to(torch.float32)
features = torch.tensor(np.loadtxt("63.txt", max_rows = size * 63).reshape(size, 63)).to(torch.float32)
val_size = 3000
val_labels = torch.tensor(np.loadtxt("val_B0.txt", max_rows = val_size * 5).reshape(val_size, 25)).to(torch.float32)
val_features = torch.tensor(np.loadtxt("val_63.txt", max_rows = val_size * 63).reshape(val_size, 63)).to(torch.float32)

input_num, output_num = 63, 25
dnn_net = nn.Sequential(
    nn.Linear(input_num, 128), nn.Sigmoid(),
    nn.Linear(128, 128, 128), nn.Sigmoid(),
    nn.Linear(128, 128, 128), nn.Sigmoid(),
    nn.Linear(128, 128, 128), nn.Sigmoid(),
    nn.Linear(128, 128, 128), nn.Sigmoid(),
    nn.Linear(128, 128, 128), nn.Sigmoid(),
    nn.Linear(128, 128, 128), nn.Sigmoid(),
    nn.Linear(128, output_num)
)

res_net = nn.Sequential(
    nn.Linear(input_num, 128), nn.Sigmoid(),
    Residual(128, 128, 128), nn.Sigmoid(),
    Residual(128, 128, 128), nn.Sigmoid(),
    Residual(128, 128, 128), nn.Sigmoid(),
    nn.Linear(128, output_num)
)

batch_size, lr, epochs = 100, 0.3, 1000
X = torch.rand(size = (batch_size, input_num))
for layer in dnn_net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape, )

# read batch_size sample
train_iter = load_data((features, labels), batch_size)
valid_iter = load_data((val_features, val_labels), batch_size)

# init weights norm
dnn_net.apply(init_weights)
# L2 loss
loss = nn.MSELoss()
# sgd train
trainer = torch.optim.SGD(dnn_net.parameters(), lr = lr)


def train_loss(net, data_iter, loss):
    """ Evaluate the loss of a model while training."""
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = d2l.reshape(y, out.shape)
        l = loss(out, y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
        metric.add(d2l.reduce_sum(l), d2l.size(l))
    return metric[0] / metric[1]


def validation_loss(net, data_iter, loss):
    """ Evaluate the loss of a model on the given dataset."""
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        # print("comapre", X, torch.gt(y, torch.gt(out, ave).to(torch.float32)))
        y = d2l.reshape(y, out.shape)
        l = loss(out, y)
        metric.add(d2l.reduce_sum(l), d2l.size(l))
    return metric[0] / metric[1]


ave = d2l.torch.ones((batch_size, output_num)) / 2
image = d2l.Animator(xlabel = 'epoch', xlim = [1, epochs], ylim = [0.01, 0.3],
                     legend = ['train loss', 'valid loss'])
for epoch in range(epochs):
    image.add(epoch + 1, (train_loss(dnn_net, train_iter, loss),
                          validation_loss(dnn_net, valid_iter, loss)))
    l = loss(dnn_net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
d2l.plt.show()

torch.save(dnn_net.state_dict(), "rev_DNN.pth")
