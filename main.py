
import argparse
import sys
import os
import csv
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

# Choose experiment here
import experiments.labeled_mnist as exp

# Methods
from methods.disconet import *
from methods.tent import *
from methods.uda import *

# PARAMETERS

parser = argparse.ArgumentParser(description='Domain Adaptation Experiment')

# Reproducibility
parser.add_argument('--seed', type=int, default=1, help='random seed')

parser.add_argument('--method', type=str, default='disconet', choices={'disconet', 'tent', 'uda'}, help='method used for domain adaptation')
parser.add_argument('--val_type', type=str, default='single', choices={'none', 'single', 'random'}, help='stripe type for validation data')

parser.add_argument('--batch_size', type=int, default=128, help='batch size used during training/testing')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--epoch_step', type=int, default=1, help='number of epochs between validation checkpoints')

parser.add_argument('--c_lr', type=float, default=1e-3, help='learning rate for classifier')
parser.add_argument('--d_lr', type=float, default=1e-3, help='learning rate for discriminators')
parser.add_argument('--g_lr', type=float, default=1e-3, help='learning rate for generators')

parser.add_argument('--jsd_lambda', type=float, default=1, help='scale for JSD consistency loss')
parser.add_argument('--g_lambda', type=float, default=1, help='scale for generator loss')
parser.add_argument('--rec_lambda', type=float, default=1, help='scale for reconstruction loss')

parser.add_argument('--save_images', action='store_true', help='saves generated/reconstructed images at each checkpoint')
parser.add_argument('--save_model', action='store_true', help='saves model after training')
parser.set_defaults(save_images=False, save_model=False)

args = parser.parse_args()

setting = '-'.join(sys.argv[1:]).replace('---', '--').replace('--', '-')

if args.save_images:
    os.mkdir('results/data/' + setting)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

params = exp.get_params(args)

Trainer = (DiscoTrainer if args.method == 'disconet'
                        else TentTrainer if args.method == 'tent' else UdaTrainer)

trainer = Trainer(params, setting)

trainer.train(n_epochs=args.n_epochs, epoch_step=args.epoch_step)
trainer.write_results()

if args.save_model:
    trainer.write_model()
