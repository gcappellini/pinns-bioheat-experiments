import os
import numpy as np
from omegaconf import OmegaConf
import hydra
# import common as co
import plots as pp
import coeff_calc as cc
from scipy import integrate
import utils as uu
import time
import torch.nn as nn
import deepxde as dde
import torch
from common import set_run, generate_config_hash
# import matlab.engine
import utils as uu 
import common as co

np.random.seed(237)

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
conf_dir = os.path.join(src_dir, "configs")
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
models = os.path.join(git_dir, "models")
os.makedirs(tests_dir, exist_ok=True)


config = OmegaConf.load(f"{conf_dir}/config_run.yaml")

file_path = f"{src_dir}/data/vessel/20240930_3.txt"
timeseries_data = uu.load_measurements(file_path)
# print(timeseries_data.keys())

meas = getattr(config.experiment_type, "meas_cool_1")
dictionary = uu.extract_entries(timeseries_data, meas.start_min*60, meas.end_min*60, keys_to_extract={42:'42', 43:'43', 44:'44', 45:'gt1', 46:'46', 48:'48', 49:'49', 24: 'y2', 12:"12"})
# print(dictionary.columns.tolist())
labels = dictionary.columns.tolist()[1:]

x = [dictionary['t']]*len(labels)
y = [dictionary[lal] for lal in labels]

pp.plot_generic(x=x, y=y, title="Test", xlabel="Time (s)", ylabel="Temperature (°C)", filename=f"{tests_dir}/visualize_measurements.png", legend_labels=labels)


