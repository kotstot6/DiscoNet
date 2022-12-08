
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Function
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from experiments.data.mnist_m import *

def get_params(args):

    params = {}

    transform_data = transforms.Compose([transforms.ToTensor()])

    def annotate_data(data, stripe='ordered'): #Stripe = [none, single, ordered, random]

        def get_label_dists(labels, data_size, n_classes=10, trials=5, weight=0.9):

            probs = [weight] + [(1 - weight) / (n_classes - 1)] * (n_classes - 1)
            samples = np.random.choice(n_classes, (data_size, trials), p=probs)
            label_dists = np.zeros((data_size, n_classes))

            for i in range(n_classes):
                label_dists[:, i] = np.mean(samples == i, axis=1)

            if stripe == 'ordered':
                for i in range(n_classes):
                    ld_class = label_dists[labels == i,:]
                    ld_class[:, [0, i]] = ld_class[:, [i, 0]]
                    label_dists[labels == i, :] = ld_class
            elif stripe == 'random':
                for i in range(label_dists.shape[0]):
                    np.random.shuffle(label_dists[i,:])

            return torch.Tensor(label_dists)

        if stripe == 'none':
            return data

        labels = np.array([label for _, label in data])
        label_dists = get_label_dists(labels, len(data))

        stripe_indices = [1,2,3,4,5,-6,-5,-4,-3,-2]
        for i in range(len(data)):
            image, label = data[i]
            image[0,:,stripe_indices] = torch.clip(image[0,:,stripe_indices] + label_dists[i,:], min=0, max=1)
            data[i] = (image, label)

        return data

    transform_data = transforms.Compose([transforms.ToTensor()])

    train_data = list(datasets.MNIST('data', download=True, train=True, transform=transform_data))
    val_data = list(datasets.MNIST('data', download=True, train=False, transform=transform_data))

    A_train_data = annotate_data(train_data[:len(train_data) // 2])
    B_train_data = annotate_data(train_data[len(train_data) // 2:], stripe=args.val_type)
    A_val_data = annotate_data(val_data[:len(val_data) // 2])
    B_val_data = annotate_data(val_data[len(val_data) // 2:], stripe=args.val_type)

    params['train_loader'] = {
        'A' : DataLoader(A_train_data, batch_size=args.batch_size, shuffle=True),
        'B' : DataLoader(B_train_data, batch_size=args.batch_size, shuffle=True),
    }

    params['val_loader'] = {
        'A' : DataLoader(A_val_data, batch_size=args.batch_size, shuffle=False),
        'B' : DataLoader(B_val_data, batch_size=args.batch_size, shuffle=False),
    }

    method_params = {'disconet' : disconet_params, 'tent' : tent_params, 'uda' : uda_params}
    params.update(method_params[args.method](args))

    return params

def disconet_params(args):

    params = {}

    params['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    class Generator(nn.Module):

        def __init__(self):

            super(Generator, self).__init__()

            self.conv1 = nn.Sequential(nn.Conv2d(1, 8, 3, stride=2, padding=1, bias=False), nn.LeakyReLU(0.2, inplace=True))
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
            self.deconv3 = nn.Sequential(nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1, bias=False))

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
                nn.Conv2d(1, 8, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(8, 16, 3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(16, 32, 3, stride=2, padding=0),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Flatten(start_dim=1),
                nn.Dropout(0.1),
                nn.Linear(3*3*32, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.1),
                nn.Linear(128,1),
            )

        def forward(self, x):
            return self.main(x).reshape(-1)

    class Classifier(nn.Module):

        def __init__(self):

            super(Classifier, self).__init__()

            self.main = nn.Sequential(
                nn.Conv2d(1, 8, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(8, 16, 3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(16, 32, 3, stride=2, padding=0),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Flatten(start_dim=1),
                nn.Dropout(0.1),
                nn.Linear(3*3*32, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.1),
                nn.Linear(128,10)
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
        'c' : optim.Adam(params['models']['c'].parameters(), lr=args.c_lr),
        'g_AB' : optim.Adam(params['models']['g_AB'].parameters(), betas=(0.8,0.999), lr=args.g_lr, weight_decay=1e-4),
        'g_BA' : optim.Adam(params['models']['g_BA'].parameters(), betas=(0.8,0.999), lr=args.g_lr, weight_decay=1e-4),
        'd_A' : optim.Adam(params['models']['d_A'].parameters(), betas=(0.8,0.999), lr=args.d_lr),
        'd_B' : optim.Adam(params['models']['d_B'].parameters(), betas=(0.8,0.999), lr=args.d_lr)
    }

    params['lambdas'] = torch.Tensor([1,args.jsd_lambda,args.jsd_lambda,args.jsd_lambda])

    def image_saver(A, B, A_fake, B_fake, A_rec, B_rec, epoch, setting):

        fig = plt.figure(figsize=(8,6))
        grid = ImageGrid(fig, 111, nrows_ncols=(3, 4), axes_pad=0.1)
        for j in range(2):
            grid[j].imshow(A[j,:].reshape(28,28), cmap='gray')
            grid[2+j].imshow(B[j,:].reshape(28,28), cmap='gray')
            grid[4+j].imshow(B_fake[j,:].reshape(28,28), cmap='gray')
            grid[6+j].imshow(A_fake[j,:].reshape(28,28), cmap='gray')
            grid[8+j].imshow(A_rec[j,:].reshape(28,28), cmap='gray')
            grid[10+j].imshow(B_rec[j,:].reshape(28,28), cmap='gray')

        plt.title('Epoch ' + str(epoch))
        plt.savefig('results/data/' +  setting + '/epoch-' + str(epoch) + '.png')
        plt.clf()
        plt.close()

    params['image_saver'] = image_saver if args.save_images else None

    return params

def tent_params(args):

    params = {}

    params['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    class Classifier(nn.Module):

        def __init__(self):

            super(Classifier, self).__init__()

            self.main = nn.Sequential(
                nn.Conv2d(1, 8, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(8, 16, 3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(16, 32, 3, stride=2, padding=0),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Flatten(start_dim=1),
                nn.Dropout(0.1),
                nn.Linear(3*3*32, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.1),
                nn.Linear(128,10)
            )

        def forward(self, x):
            return self.main(x)

    params['model'] = Classifier().to(params['device'])
    params['model'].load_state_dict(torch.load('models/-method-disconet-jsd_lambda-0-n_epochs-50-save_model.pt'))
    params['optimizer'] = {'type' : optim.Adam, 'lr' : args.c_lr}

    return params

def uda_params(args):

    params = {}

    params['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    class ReverseLayerF(Function):

        @staticmethod
        def forward(ctx, x, alpha):
            ctx.alpha = alpha
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            output = grad_output.neg() * ctx.alpha
            return output, None

    class Model(nn.Module):

        def __init__(self):
            super(Model, self).__init__()

            self.feature_extractor = nn.Sequential(
                nn.Conv2d(1, 8, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(8, 16, 3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(16, 32, 3, stride=2, padding=0),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Flatten(start_dim=1))

            self.class_classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(3*3*32, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.1),
                nn.Linear(128,10),
                nn.LogSoftmax(dim=1)
            )

            self.domain_classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(3*3*32, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.1),
                nn.Linear(128,64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.1),
                nn.Linear(64,2),
                nn.LogSoftmax(dim=1)
            )

        def forward(self, x, alpha):
            feature = self.feature_extractor(x)
            class_output = self.class_classifier(feature)
            reverse_feature = ReverseLayerF.apply(feature, alpha)
            domain_output = self.domain_classifier(reverse_feature)

            return class_output, domain_output

    params['model'] = Model().to(params['device'])
    params['optimizer'] = optim.Adam(params['model'].parameters(), lr=args.c_lr)

    return params
