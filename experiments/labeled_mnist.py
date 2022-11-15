
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import random

import matplotlib.pyplot as plt

def annotate_data(data, swap=True):

    def get_label_dists(labels, data_size, n_classes=10, trials=5, weight=0.9):

        probs = [weight] + [(1 - weight) / (n_classes - 1)] * (n_classes - 1)

        samples = np.random.choice(n_classes, (data_size, trials), p=probs)

        label_dists = np.zeros((data_size, n_classes))

        for i in range(n_classes):
            label_dists[:, i] = np.mean(samples == i, axis=1)

        if swap:
            for i in range(n_classes):
                ld_class = label_dists[labels == i,:]
                ld_class[:, [0, i]] = ld_class[:, [i, 0]]
                label_dists[labels == i, :] = ld_class

        return torch.Tensor(label_dists)

    labels = np.array([label for _, label in data])

    label_dists = get_label_dists(labels, len(data))

    for i in range(len(data)):
        image, label = data[i]
        image[0,:,-10:] = torch.clip(image[0,:,-10:] + label_dists[i,:], min=0, max=1)
        data[i] = (image, label)

    return data

transform_data = transforms.Compose([transforms.ToTensor()])

train_data = list(datasets.MNIST('data', download=True, train=True, transform=transform_data))
val_data = list(datasets.MNIST('data', download=True, train=False, transform=transform_data))

A_train_data = annotate_data(train_data[:len(train_data) // 2])
B_train_data = annotate_data(train_data[len(train_data) // 2:], swap=False)
A_val_data = annotate_data(val_data[:len(val_data) // 2])
B_val_data = annotate_data(val_data[len(val_data) // 2:], swap=False)

batch_size = 128

train_loader = {
    'A' : DataLoader(A_train_data, batch_size=batch_size, shuffle=True),
    'B' : DataLoader(B_train_data, batch_size=batch_size, shuffle=True),
}

val_loader = {
    'A' : DataLoader(A_val_data, batch_size=batch_size, shuffle=True),
    'B' : DataLoader(B_val_data, batch_size=batch_size, shuffle=True),
}

class Generator(nn.Module):

    def __init__(self):

        super(Generator, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=0), nn.ReLU())

        self.lin = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(3*3*32, 128),
            nn.ReLU(),
            nn.Linear(128, 3*3*32),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(32,3,3))
        )

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0), nn.BatchNorm2d(16), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(8), nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1))

        self.sig = nn.Sigmoid()

        """
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(3*3*32, 128),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 3*3*32),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(32,3,3)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        """

    def forward(self, x):

        xc1 = self.conv1(x)
        xc2 = self.conv2(xc1)
        xc3 = self.conv3(xc2)

        #print('xc1', xc1.shape, 'xc2', xc2.shape, 'xc3', xc3.shape)

        xd1 = self.deconv1(self.lin(xc3)) + xc2
        xd2 = self.deconv2(xd1) + xc1
        xd3 = self.deconv3(xd2) + x

        #print('xd1', xd1.shape, 'xd2', xd2.shape, 'xd3', xd3.shape)

        return self.sig(xd3)

class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(3*3*32, 128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).reshape(-1)

class Classifier(nn.Module):

    def __init__(self):

        super(Classifier, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(3*3*32, 128),
            nn.ReLU(),
            nn.Linear(128,10),
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
    'c' : optim.Adam(models['c'].parameters(), lr=1e-3),
    'g_AB' : optim.Adam(models['g_AB'].parameters(), lr=1e-3, weight_decay=1e-3),
    'g_BA' : optim.Adam(models['g_BA'].parameters(), lr=1e-3, weight_decay=1e-3),
    'd_A' : optim.Adam(models['d_A'].parameters(), lr=1e-3),
    'd_B' : optim.Adam(models['d_B'].parameters(), lr=1e-3)
}

lambdas = torch.Tensor([10,1,1,1])
