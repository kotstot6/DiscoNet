
from itertools import product
import os
import csv
import argparse

parser = argparse.ArgumentParser(description='Run Domain Adaptation Experiment')
parser.add_argument('--sbatch', action='store_true', help='run parameter grid on sbatch')
parser.set_defaults(sbatch=False)
args = parser.parse_args()

# Prepare results directory
if os.path.isdir('results'):
    os.system('rm -rf results')

os.mkdir('results')
os.mkdir('results/outputs')
os.mkdir('results/data')

with open('results/metrics.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Setting', 'final_cA', 'final_cB', 'final_dA', 'final_dB',
                                            'all_cA', 'all_cB', 'all_dA', 'all_dB'])

# String of args for single use
single_params = '--seed 1 --save_images'

# Grid of hyperparameters for sbatch
grid = {
    'g_lr' : [1e-4, 1e-3, 1e-2],
    'd_lr' : [1e-4, 1e-3, 1e-2],
    'jsd_lambda' : [0.2,1,5],
    'g_lambda' : [0.2,1,5],
    'rec_lambda' : [0.2,1,5],
}

# Utility function
def make_sbatch_params(grid):

    trials = [ { p : t for p, t in zip(grid.keys(), trial) }
                    for trial in list(product(*grid.values())) ]

    def trial_to_args(trial):
        arg_list = ['--' + param + ' ' + str(val) if type(val) != type(True)
                else '--' + param if val else '' for param, val in trial.items()]
        return ' '.join(arg_list)

    sbatch_params = [trial_to_args(trial) for trial in trials]

    return sbatch_params

sbatch_params = make_sbatch_params(grid)

if args.sbatch:

    print(len(sbatch_params), 'jobs will be submitted.')

    for params in sbatch_params:
        os.system('sbatch run_experiment.sh \'' + params + '\'')

else:

    print('Interactive mode.')
    os.system('python3 main.py ' + single_params)
