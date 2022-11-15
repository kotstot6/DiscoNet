
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from experiments.data.mnist_m import *

def get_params(args):

    params = {}

    params['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform_data = transforms.Compose([transforms.ToTensor()])

    def resize_data(data):
        data = list(data)

        for i in range(len(data)):
            image, label = data[i]
            data[i] = (image.expand(3,28,28), label)

        return data

    A_train_data = resize_data(datasets.MNIST('data', download=True, train=True, transform=transform_data))
    B_train_data = list(MNISTM('data', download=True, train=True, transform=transform_data))
    A_val_data = resize_data(datasets.MNIST('data', download=True, train=False, transform=transform_data))
    B_val_data = list(MNISTM('data', download=True, train=False, transform=transform_data))

    A_train_data = A_train_data[:len(A_train_data) // 2]
    B_train_data = B_train_data[len(A_train_data) // 2:]
    A_val_data = A_val_data[:len(A_val_data) // 2]
    B_val_data = B_val_data[len(A_val_data) // 2:]

    params['train_loader'] = {
        'A' : DataLoader(A_train_data, batch_size=args.batch_size, shuffle=True),
        'B' : DataLoader(B_train_data, batch_size=args.batch_size, shuffle=True),
    }

    params['val_loader'] = {
        'A' : DataLoader(A_val_data, batch_size=args.batch_size, shuffle=True),
        'B' : DataLoader(B_val_data, batch_size=args.batch_size, shuffle=True),
    }

    class Generator(nn.Module):

        def __init__(self):

            super(Generator, self).__init__()

            self.conv1 = nn.Sequential(nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=False), nn.LeakyReLU(0.2, inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(8, 16, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(16), nn.LeakyReLU(0.2, inplace=True))
            self.conv3 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=0, bias=False), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True))

            self.lin = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(3*3*32, 128),
                nn.ReLU(),
                nn.Linear(128, 3*3*32),
                nn.ReLU(),
                nn.Unflatten(dim=1, unflattened_size=(32,3,3))
            )

            self.deconv1 = nn.Sequential(nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0, bias=False), nn.BatchNorm2d(16), nn.ReLU(True))
            self.deconv2 = nn.Sequential(nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(True))
            self.deconv3 = nn.Sequential(nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1, bias=False))

            self.sig = nn.Sigmoid()

        def forward(self, x):

            xc1 = self.conv1(x)
            xc2 = self.conv2(xc1)
            xc3 = self.conv3(xc2)

            xd1 = self.deconv1(self.lin(xc3) + xc3) + xc2
            xd2 = self.deconv2(xd1) + xc1
            xd3 = self.deconv3(xd2) + x

            return self.sig(xd3)

    class Discriminator(nn.Module):

        def __init__(self):

            super(Discriminator, self).__init__()

            self.main = nn.Sequential(
                nn.Conv2d(3, 8, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(8, 16, 3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(16, 32, 3, stride=2, padding=0),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Flatten(start_dim=1),
                nn.Linear(3*3*32, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(128,1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.main(x).reshape(-1)

    class Classifier(nn.Module):

        def __init__(self):

            super(Classifier, self).__init__()

            self.main = nn.Sequential(
                nn.Conv2d(3, 8, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(8, 16, 3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(16, 32, 3, stride=2, padding=0),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Flatten(start_dim=1),
                nn.Linear(3*3*32, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(128,10),
                nn.Softmax(dim=1)
            )

        def forward(self, x):
            return self.main(x)

    params['models'] = {
        'c' : Classifier().to(params['device']),
        'g_AB' : Generator().to(params['device']),
        'g_BA' : Generator().to(params['device']),
        'd_A' : Discriminator().to(params['device']),
        'd_B' : Discriminator().to(params['device'])
    }

    params['optimizers'] = {
        'c' : optim.Adam(params['models']['c'].parameters(), lr=args.d_lr),
        'g_AB' : optim.Adam(params['models']['g_AB'].parameters(), betas=(0.8,0.999), lr=args.g_lr, weight_decay=1e-4),
        'g_BA' : optim.Adam(params['models']['g_BA'].parameters(), betas=(0.8,0.999), lr=args.g_lr, weight_decay=1e-4),
        'd_A' : optim.Adam(params['models']['d_A'].parameters(), betas=(0.8,0.999), lr=args.d_lr),
        'd_B' : optim.Adam(params['models']['d_B'].parameters(), betas=(0.8,0.999), lr=args.d_lr)
    }

    params['lambdas'] = torch.Tensor([1,args.jsd_lambda,args.g_lambda,args.rec_lambda])

    def image_saver(A, B, A_fake, B_fake, A_rec, B_rec, epoch, setting):

        fig = plt.figure(figsize=(8,6))
        grid = ImageGrid(fig, 111, nrows_ncols=(3, 4), axes_pad=0.1)
        for j in range(2):
            grid[j].imshow(A[j,:].permute(1,2,0))
            grid[2+j].imshow(B[j,:].permute(1,2,0))
            grid[4+j].imshow(B_fake[j,:].permute(1,2,0))
            grid[6+j].imshow(A_fake[j,:].permute(1,2,0))
            grid[8+j].imshow(A_rec[j,:].permute(1,2,0))
            grid[10+j].imshow(B_rec[j,:].permute(1,2,0))

        plt.title('Epoch ' + str(epoch))
        plt.savefig('results/data/' +  setting + '/epoch-' + str(epoch) + '.png')
        plt.clf()
        plt.close()

    params['image_saver'] = image_saver if args.save_images else None

    return params
