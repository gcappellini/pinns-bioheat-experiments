import deepxde as dde
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import wandb
import json
from scipy.interpolate import interp1d
from scipy import integrate
import pickle
import pandas as pd
import hashlib

dde.config.set_random_seed(200)

# device = torch.device("cpu")
device = torch.device("cuda")

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)


models = os.path.join(git_dir, "models")
os.makedirs(models, exist_ok=True)

figures = os.path.join(tests_dir, "figures")
os.makedirs(figures, exist_ok=True)


prj_figs, run_figs = [None]*2

def set_prj(prj):
    global prj_figs

    prj_figs = os.path.join(figures, prj)
    os.makedirs(prj_figs, exist_ok=True)

    return prj_figs


def set_run(run):
    global prj_figs, run_figs

    run_figs = os.path.join(prj_figs, run)
    os.makedirs(run_figs, exist_ok=True)

    return run_figs

def read_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)
    return data


def generate_config_hash(config_data):
    config_string = json.dumps(config_data, sort_keys=True)  # Sort to ensure consistent ordering
    config_hash = hashlib.md5(config_string.encode()).hexdigest()  # Create a unique hash
    return config_hash



def write_json(data, filepath):
    def convert_to_serializable(obj):
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable_data = {k: convert_to_serializable(v) for k, v in data.items()}

    with open(filepath, 'w') as file:
        json.dump(serializable_data, file, indent=4)