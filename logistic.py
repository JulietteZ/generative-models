#%%
import torch
from torch import nn
from torch import optim
import numpy as np
from matplotlib import pyplot as plt

#%%
# logistic regression from scratch
def logistic_regression(x, w, b):
    ''' params: x: (m, n)
        params: w: (n, l=1)
        params: b: float
    '''
    z = np.matmul(x, w) + b
    a = sigmoid(z)
    return a

def sigmoid(z):
    return 1. / (1. + np.exp(- z))

def bce(pred, y):
    m = pred.shape[0]
    return - 1. / m * sum(y * np.log(pred) + (1 - y) * np.log(1 - pred))

def plot(x, y, w, b):
    # plot the real result against predicted result by logistic regression
    plt.figure()
    plt.plot(x, y, 'or', label='real result')
    pred = logistic_regression(x, w, b)
    plt.plot(x, pred, '*b', label='predicted result')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')

def gradient_descent():
    x = np.array([[i] * 5 for i in np.arange(5)])
    y = np.array([0,0,1,1,1]).reshape(5, -1)
    # random initialization of weight matrix and bias
    w = np.random.randn(5, 1)
    b = np.random.randn(1)
    m = x.shape[0]
    n = x.shape[1]
    alpha = 1e-2
    epochs = 10000
    print(f'initial w {w}, b {b}')
    plot(x, y, w, b)
    
    for epoch in range(epochs):
        pred = logistic_regression(x, w, b)
        loss = bce(pred, y)
        dj_dw = np.sum((pred - y) * x, axis=0).reshape(n, 1) / m
        dj_db = np.sum((pred - y), axis=0) / m
        w -= alpha * dj_dw
        b -= alpha * dj_db
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, loss {loss}')
    print(f'final w {w}, b {b}')
    plot(x, y, w, b)

gradient_descent()


#%%
# logitic regression using pytorch
class LogisticReg(nn.Module):

    def __init__(self, input_size):
        # class attributes, will automatically treated as parameters 
        # if they are themselves nn objects or if they are tensors 
        # wrapped in nn.Parameter which are initialized with the class.
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1),
        )

    def forward(self, x):
        return self.model(x)

# %%

if __name__ == '__main__':
    # x = torch.tensor([1.,2.,3.,4.,5.]).view(5, -1)
    x = torch.tensor([[i] * 4 for i in np.arange(5)])
    x = x.to(torch.float32)
    print(x)
    y = torch.zeros(5).view(5, -1)
    # z = torch.randn(2,5)
    # print(z, z.size(0))
    y[2:] = 1 - y[2:]
    print(y.dtype, y)

    k = y == 1
    print(k.to(torch.float32))

    epochs = 10
    lr = 1e-3
    betas = (0.9, 0.99)

    criterion = nn.BCEWithLogitsLoss()
    logistic = LogisticReg(4)
    print(list(logistic.named_parameters()))
    optimizer = torch.optim.Adam(logistic.parameters(), lr=lr, betas=betas)

    for epoch in range(epochs):
        pred = logistic(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch}, loss {loss.item()}')

# %%
