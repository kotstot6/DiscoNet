
# DiscoNet :mirror_ball: Towards Mitigating Shortcut Learning with Cross-Domain Regularization

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)


## Authors

- Kyle Otstot ([@kotstot6](https://www.github.com/kotstot6))
- John Kevin Cava ([@jcava](https://www.github.com/jcava))


## Technologies

**Backend:** Python, PyTorch, NumPy

**Cluster Environment:** Slurm, SBATCH


## Architecture

[![Hn70eOF.md.png](https://iili.io/Hn70eOF.md.png)](https://freeimage.host/i/Hn70eOF)
## Experiment Results

#### Dataset: Striped-MNIST

#### Target #1:  No stripes `--val_type none`

| Method | Domain | Accuracy (mean ± std) |
| :--------: | :-------: | :--------: |
| TENT | S | 99.7 ± 0.0 |
| TENT | T | 72.7 ± 0.1 |
| UDA | S | 98.1 ± 0.2 |
| UDA | T | 99.3 ± 0.0 |
| DiscoNet | S | **99.3 ± 0.4** |
| DiscoNet | T | **99.5 ± 0.1** |

#### Target #2:  Single stripe `--val_type single`

| Method | Domain | Accuracy (mean ± std) |
| :--------: | :-------: | :--------: |
| TENT | S | 59.7 ± 31.3 |
| TENT | T | 10.4 ± 0.0 |
| UDA | S | 98.0 ± 0.2 |
| UDA | T | 99.2 ± 0.1 |
| DiscoNet | S | **99.4 ± 0.3** |
| DiscoNet | T | **99.4 ± 0.2** |

#### Target #3:  Random stripes `--val_type random`

| Method | Domain | Accuracy (mean ± std) |
| :--------: | :-------: | :--------: |
| TENT | S | 99.9 ± 0.0 |
| TENT | T | 10.1 ± 0.3 |
| UDA | S | **99.8 ± 0.1** |
| UDA | T | 10.0 ± 0.2 |
| DiscoNet | S | 99.6 ± 0.7 |
| DiscoNet | T | **27.9 ± 40.0** |

## Run Experiment

Clone the project

```bash
  git clone https://link-to-project
```

Go to the project directory

```bash
  cd /path/to/DiscoNet
```

#### Option 1: Run Locally

```bash
  python3 main.py ( ... parameters ...)
```

| Parameter | Type | Default | Description |
| :-------- | :------- | :-------- | :----- |
|`--seed`|`int`| `1` |random seed for reproducibility|
|`--method`|`str`| `'disconet'` |method used for domain adaptation|
|`--val_type`|`str`| `'single'` |stripe type for validation data|
|`--batch_size`|`int`| `128` | batch size used during training/testing |
|`--n_epochs`|`int`| `100` |number of epochs for training|
|`--epoch_step`|`int`| `1` |number of epochs between validation checkpoints|
|`--c_lr`|`float`| `1e-3`|learning rate for classifier|
|`--d_lr`|`float`| `1e-3` |learning rate for discriminators|
|`--g_lr`|`float`| `1e-3` |learning rate for generators|
|`--jsd_lambda`|`float`| `1`|scale for JSD consistency loss|
|`--g_lambda`|`float`| `1`|scale for generator loss|
|`--rec_lambda`|`float`| `1`|scale for reconstruction loss|
|`--save_images`| | `False`|saves generated/reconstructed images at each checkpoint|
|`--save_model`| | `False`|saves model after training|

#### Option 2: Slurm Environment

In `run_experiment.py`, adjust `grid` dictionary used for hyperparameter grid search. Example:

```python
grid = {
    'method' : ['disconet', 'tent', 'uda'],
    'val_type' : ['random'],
    'seed' : [1,2,3,4,5]
}
```

Adjust the `run_experiment.sh` SBATCH script for use.

Lastly, run the following command:

```bash
python3 run_experiment.py --sbatch
```
