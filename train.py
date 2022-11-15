
import csv
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

class Trainer:

    def __init__(self, params, setting):

        self.device = params['device']

        self.train_loader = params['train_loader']
        self.val_loader = params['val_loader']
        self.models = params['models']
        self.optimizers = params['optimizers']
        self.lambdas = params['lambdas'].to(self.device)
        self.image_saver = params['image_saver']
        self.setting = setting

        self.all_cA = []
        self.all_cB = []
        self.all_dA = []
        self.all_dB = []

    def disc_loss(self, disc_key, X, X_fake, k=1):

        input = torch.concat((X, X_fake), dim=0)
        labels = torch.Tensor([1] * X.shape[0] + [0] * X_fake.shape[0]).to(self.device)
        flip_labels = torch.Tensor([0] * X.shape[0] + [1] * X_fake.shape[0]).to(self.device)
        d_criterion = nn.BCELoss().to(self.device)
        g_criterion = nn.BCELoss().to(self.device)

        for i in range(k):

            self.optimizers[disc_key].zero_grad()
            output = self.models[disc_key](input.detach())
            d_loss = d_criterion(output, labels)
            d_loss.backward()
            self.optimizers[disc_key].step()

        output = self.models[disc_key](input)
        g_loss = g_criterion(output, flip_labels)

        return g_loss

    def jsd(self, P1, P2):

        P_mix = (P1 + P2) / 2

        KL1 = torch.sum(P1 * torch.log(P1 / P_mix), dim=1)
        KL2 = torch.sum(P2 * torch.log(P2 / P_mix), dim=1)

        return torch.mean(KL1 + KL2)

    def class_loss(self, X, Y_fake, labels=None):

        input = torch.concat((X, Y_fake), dim=0)

        output = self.models['c'](input)

        P1, P2 = output[:X.shape[0],:], output[X.shape[0]:, :]

        jsd_loss = self.jsd(P1, P2)

        criterion = nn.CrossEntropyLoss().to(self.device)

        c_loss = torch.Tensor([0]).to(self.device) if labels is None else criterion(P1, labels)

        return c_loss, jsd_loss

    def rec_loss(self, X, X_fake):
        return torch.mean((X - X_fake) ** 2)

    def train_loop(self, X, Y, g_XY_key, g_YX_key, d_Y_key, labels=None):

        for model in self.models.values():
            model.train()

        Y_fake = self.models[g_XY_key](X)

        g_loss = self.disc_loss(d_Y_key, Y, Y_fake)

        c_loss, jsd_loss = self.class_loss(X, Y_fake, labels=labels)

        X_fake = self.models[g_YX_key](Y_fake)

        r_loss = self.rec_loss(X, X_fake)

        losses = torch.concat([loss.reshape(1,-1) for loss in [c_loss, jsd_loss, g_loss, r_loss]], dim=1)

        return losses

    def train_epoch(self, epoch):

        all_losses = None

        for (A, labels), (B, _) in tqdm(zip(self.train_loader['A'], self.train_loader['B']), total=len(self.train_loader['A'])):

            A, B = A.to(self.device), B.to(self.device)
            labels = labels.to(self.device)

            for optimizer in self.optimizers.values():
                optimizer.zero_grad()

            losses = self.train_loop(A, B, 'g_AB', 'g_BA', 'd_B', labels=labels)
            losses += self.train_loop(B, A, 'g_BA', 'g_AB', 'd_A', labels=None)

            total_loss = torch.sum(losses * self.lambdas)
            total_loss.backward()

            for name, optimizer in self.optimizers.items():
                if name not in ['d_A', 'd_B']:
                    optimizer.step()

            all_losses = losses if all_losses is None else torch.concat((all_losses, losses), dim=0)

        avg_losses = torch.mean(all_losses, dim=0)

        print('----- Epoch', epoch, '-----')
        print('Classifier loss:', float(avg_losses[0]))
        print('JSD loss:', float(avg_losses[1]))
        print('Generator loss:', float(avg_losses[2]))
        print('Reconstruction loss:', float(avg_losses[3]))
        print('Total loss:', float(torch.sum(avg_losses * self.lambdas)))
        print('------------------------')
        print()

    def evaluate(self, epoch):

        for model in self.models.values():
            model.eval()

        class_preds = { 'A' : [], 'B' : [] }
        class_labels = { 'A' : [], 'B' : [] }

        disc_preds = { 'A' : [], 'B' : [] }
        disc_labels = { 'A' : [], 'B' : [] }

        with torch.no_grad():

            for i, ((A, A_labels), (B, B_labels)) in enumerate(zip(self.val_loader['A'], self.val_loader['B'])):

                A, B = A.to(self.device), B.to(self.device)

                class_preds['A'] += self.models['c'](A).cpu().detach().argmax(dim=1).tolist()
                class_labels['A'] += A_labels.tolist()

                class_preds['B'] += self.models['c'](B).cpu().detach().argmax(dim=1).tolist()
                class_labels['B'] += B_labels.tolist()

                B_fake = self.models['g_AB'](A)
                B_all = torch.concat((B, B_fake), dim=0)

                disc_preds['B'] += self.models['d_B'](B_all).cpu().round().int().tolist()
                disc_labels['B'] += [1] * B.shape[0] + [0] * B_fake.shape[0]

                A_fake = self.models['g_BA'](B)
                A_all = torch.concat((A, A_fake), dim=0)
                disc_preds['A'] += self.models['d_A'](A_all).cpu().round().int().tolist()
                disc_labels['A'] += [1] * A.shape[0] + [0] * A_fake.shape[0]

                A_rec = self.models['g_BA'](B_fake)
                B_rec = self.models['g_AB'](A_fake)

                if i == 0 and self.image_saver is not None:
                    self.image_saver(A.cpu(), B.cpu(), A_fake.cpu(), B_fake.cpu(),
                                        A_rec.cpu(), B_rec.cpu(), epoch, self.setting)


        self.all_cA.append(np.mean(np.array(class_preds['A']) == np.array(class_labels['A'])))
        self.all_cB.append(np.mean(np.array(class_preds['B']) == np.array(class_labels['B'])))
        self.all_dA.append(np.mean(np.array(disc_preds['A']) == np.array(disc_labels['A'])))
        self.all_dB.append(np.mean(np.array(disc_preds['B']) == np.array(disc_labels['B'])))

        print('===== Classification =====')
        print('(A) acc:', self.all_cA[-1])
        print('(B) acc:', self.all_cB[-1])
        print('==========================')
        print()
        print('===== Discrimination =====')
        print('(A) acc:', self.all_dA[-1])
        print('(B) acc:', self.all_dB[-1])
        print('==========================')
        print()


    def train(self, n_epochs=500, epoch_step=1):

        for epoch in range(1, n_epochs+1):

            self.train_epoch(epoch)

            if epoch % epoch_step == 0:

                self.evaluate(epoch)

    def write_results(self):

        with open('results/metrics.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([self.setting, self.all_cA[-1], self.all_cB[-1],
                            self.all_dA[-1], self.all_dB[-1], self.all_cA,
                            self.all_cB, self.all_dA, self.all_dB])
