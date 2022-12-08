
from itertools import product
import os
import csv
import argparse

parser = argparse.ArgumentParser(description='Run Domain Adaptation Experiment')
parser.add_argument('--sbatch', action='store_true', help='run parameter grid on sbatch')
parser.add_argument('--no_reset', action='store_true', help='don\'t clear the results directory')
parser.set_defaults(sbatch=False, no_reset=False)
args = parser.parse_args()

if not args.no_reset:
    # Prepare results directory
    if os.path.isdir('results'):
        os.system('rm -rf results')

    os.mkdir('results')
    os.mkdir('results/outputs')
    os.mkdir('results/data')
    os.mkdir('results/data/test')

    with open('results/metrics.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Setting', 'final_cA', 'final_cB', 'final_dA', 'final_dB',
                                                'all_cA', 'all_cB', 'all_dA', 'all_dB'])

# String of args for single use
single_params = '--method disconet --jsd_lambda 0 --n_epochs 50 --save_model'

# Grid of hyperparameters for sbatch
grid = {
    'method' : ['disconet'],
    'val_type' : ['random'],
    'g_lr' : [1e-2],
    'jsd_lambda' : [50],
    'n_epochs' : [300],
    'seed' : list(range(1,21)),
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
