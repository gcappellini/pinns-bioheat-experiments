import deepxde as dde
import numpy as np
import os
import torch
import json
import hashlib
import logging
from omegaconf import OmegaConf
from hydra import compose
import plots as pp
import matplotlib.pyplot as plt

dde.config.set_random_seed(200)

# device = torch.device("cpu")
device = torch.device("cuda")

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
conf_dir = os.path.join(src_dir, "configs")
os.makedirs(tests_dir, exist_ok=True)

config = compose(config_name="config_run")
output_folder = f"{tests_dir}/simulation_sp"
res = []
for el in os.listdir(output_folder):
    el_path = os.path.join(output_folder, el)
    metrics = OmegaConf.load(f"{el_path}/metrics.yaml")
    pars = OmegaConf.load(f"{el_path}/.hydra/overrides.yaml")
    metrics = {key.replace("observer_4_", ""): value for key, value in metrics.items()}
    res.append([el, *pars, metrics])


# Extract parameters and metrics
x = [p[0] for p in res]
y = [p[1] for p in res]

# Iterate through all metrics keys
for metric_key in res[0][2].keys():
    z = [m[metric_key] for _, _, m in res]  # Extract metric values for the current key

    # Create 2D plot
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(x, y, c=z, cmap='viridis')
    plt.colorbar(sc, label=f'Metric Value ({metric_key})')
    plt.xlabel(f'{pars.keys()[0]}')
    plt.ylabel(f'{pars.keys()[0]}')
    plt.title(f'{metric_key}')
    plt.show()
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{metric_key}.png", dpi=120)
    plt.close()
