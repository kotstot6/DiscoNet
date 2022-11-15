
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random

def get_ring(N, c_x, c_y, radius, noise, label=0, dots=0):

    if dots > 0:
        thetas = [(2 * np.pi * i/dots) + 0.05 * torch.randn(N//dots).reshape(-1,1) for i in range(dots)]
        thetas = torch.concat(thetas, dim=0).reshape(-1)
    else:
        thetas = 2 * np.pi * torch.rand(N)

    rs = torch.randn(N) * noise + radius

    xs = rs * torch.cos(thetas) + c_x
    ys = rs * torch.sin(thetas) + c_y

    X = torch.concat((xs.reshape(-1,1), ys.reshape(-1,1)), dim=1)

    return [ (X[i,:], label) for i in range(X.shape[0])]

train_size = 15000
val_size = 5000
batch_size = 128

A_data = get_ring((train_size + val_size) // 2, -3, 5, 3, 0.1, label=0, dot=8) + get_ring((train_size + val_size) // 2, -3, 5, 2, 0.1, label=1)
B_data = get_ring((train_size + val_size) // 2, 5, -7, 3, 0.2, label=0, dot=8) + get_ring((train_size + val_size) // 2, 5, -7, 5, 0.2, label=1)

random.shuffle(A_data)
random.shuffle(B_data)

train_loader = {
    'A' : DataLoader(A_data[:train_size], batch_size=batch_size, shuffle=True),
    'B' : DataLoader(B_data[:train_size], batch_size=batch_size, shuffle=True)
}

val_loader = {
    'A' : DataLoader(A_data[train_size:], batch_size=batch_size, shuffle=True),
    'B' : DataLoader(B_data[train_size:], batch_size=batch_size, shuffle=True)
}

class Generator(nn.Module):

    def __init__(self):

        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(2,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,2)
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(2,50),
            nn.LeakyReLU(0.2),
            nn.Linear(50,50),
            nn.LeakyReLU(0.2),
            nn.Linear(50,50),
            nn.LeakyReLU(0.2),
            nn.Linear(50,50),
            nn.LeakyReLU(0.2),
            nn.Linear(50,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).reshape(-1)

class Classifier(nn.Module):

    def __init__(self):

        super(Classifier, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(2,10),
            nn.LeakyReLU(0.2),
            nn.Linear(10,10),
            nn.LeakyReLU(0.2),
            nn.Linear(10,2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.main(x)

models = {
    'c' : Classifier(),
    'g_AB' : Generator(),
    'g_BA' : Generator(),
    'd_A' : Discriminator(),
    'd_B' : Discriminator()
}

optimizers = {
    'c' : optim.Adam(models['c'].parameters(), lr=1e-2),
    'g_AB' : optim.Adam(models['g_AB'].parameters(), lr=1e-3),
    'g_BA' : optim.Adam(models['g_BA'].parameters(), lr=1e-3),
    'd_A' : optim.Adam(models['d_A'].parameters(), lr=1e-3),
    'd_B' : optim.Adam(models['d_B'].parameters(), lr=1e-3)
}

lambdas = torch.Tensor([0,0,50,1])
