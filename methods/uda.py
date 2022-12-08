
import csv
import torch
import numpy as np
from tqdm import tqdm

class UdaTrainer:

    def __init__(self, params, setting):

        self.device = params['device']

        self.train_loader = params['train_loader']
        self.val_loader = params['val_loader']

        self.model = params['model']
        self.optimizer = params['optimizer']

        self.class_criterion = torch.nn.NLLLoss()
        self.domain_criterion = torch.nn.NLLLoss()

        self.setting = setting

        self.all_cA = []
        self.all_cB = []
        self.all_dA = []
        self.all_dB = []

    def train_epoch(self, epoch, n_epochs):

        self.model.train()

        class_losses = []
        domain_losses = []

        len_loader = len(self.train_loader['A'])
        for i, ((A, labels), (B, _)) in tqdm(enumerate(zip(self.train_loader['A'], self.train_loader['B'])), total=len_loader):

            p = float(i + (epoch - 1) * len_loader) / n_epochs / len_loader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            A, B = A.to(self.device), B.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            X = torch.concat((A,B), dim=0)
            domain_labels = torch.Tensor([0] * A.shape[0] + [1] * B.shape[0]).type(torch.LongTensor).to(self.device)
            class_output, domain_output = self.model(X, alpha)

            loss_class = self.class_criterion(class_output[:A.shape[0],:], labels)
            loss_domain = self.domain_criterion(domain_output, domain_labels)
            total_loss = loss_class + loss_domain
            total_loss.backward()
            self.optimizer.step()

            class_losses.append(float(loss_class))
            domain_losses.append(float(loss_domain))

        print('----- Epoch', epoch, '-----')
        print('Class Classifier loss:', np.mean(class_losses))
        print('Domain Classifier loss:', np.mean(domain_losses))
        print('------------------------')
        print()

    def evaluate(self, epoch):

        self.model.eval()

        class_preds = { 'A' : [], 'B' : [] }
        class_labels = { 'A' : [], 'B' : [] }

        disc_preds = { 'A' : [], 'B' : [] }
        disc_labels = { 'A' : [], 'B' : [] }

        with torch.no_grad():

            for i, ((A, A_labels), (B, B_labels)) in enumerate(zip(self.val_loader['A'], self.val_loader['B'])):

                A, B = A.to(self.device), B.to(self.device)

                class_output_A, domain_output_A = self.model(A, 1)
                class_preds['A'] += class_output_A.cpu().detach().argmax(dim=1).tolist()
                class_labels['A'] += A_labels.tolist()
                disc_preds['A'] += domain_output_A.cpu().detach().argmax(dim=1).tolist()
                disc_labels['A'] += [0] * A.shape[0]

                class_output_B, domain_output_B = self.model(B, 1)
                class_preds['B'] += class_output_B.cpu().detach().argmax(dim=1).tolist()
                class_labels['B'] += B_labels.tolist()
                disc_preds['B'] += domain_output_B.cpu().detach().argmax(dim=1).tolist()
                disc_labels['B'] += [1] * B.shape[0]

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
            self.train_epoch(epoch, n_epochs)

            if epoch % epoch_step == 0:
                self.evaluate(epoch)

    def write_results(self):

        with open('results/metrics.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([self.setting, self.all_cA[-1], self.all_cB[-1],
                            self.all_dA[-1], self.all_dB[-1], self.all_cA,
                            self.all_cB, self.all_dA, self.all_dB])

    def write_model(self):
        torch.save(self.model.state_dict(), 'models/' + self.setting + '.pt')
