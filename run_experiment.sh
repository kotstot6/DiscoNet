#!/bin/bash

##SBATCH -N 1
#SBATCH -p htcgpu
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH -q normal                     # Run job under wildfire QOS queue
#SBATCH -t 00-4:00                     # wall time (D-HH:MM)
#SBATCH -o results/outputs/%j.out                  # STDOUT (%j = JobId)
#SBATCH -e results/outputs/%j.err                  # STDERR (%j = JobId)
#SBATCH --mail-type=ALL                 # Send a notification when a job starts, stops, or fails
#SBATCH --mail-user=%u@asu.edu     # send-to address

module load anaconda/py3
source activate pytorch_env
python3 main.py $1
conda deactivate
