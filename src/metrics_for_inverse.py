import pandas as pd 
import wandb
import os
import yaml
import csv
import utils as uu
from omegaconf import OmegaConf
import numpy as np
import glob

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
conf_dir = os.path.join(src_dir, "configs")
models = os.path.join(git_dir, "models")
tests_dir = os.path.join(git_dir, "tests")

base_out_path = '/Users/guglielmocappellini/Desktop/research/code/pinns-bioheat-experiments/tests/cooling_inverse'
matlab_dir = f"{tests_dir}/cooling_ground_truth_5e-04"
conf = OmegaConf.load(f"{conf_dir}/config_run.yaml")
props = conf.model_properties
pars = conf.model_parameters
folders = glob.glob(f"{base_out_path}/*")
folders = [folder for folder in folders if 'meas' not in folder]

for folder in folders:
    run_name = os.path.basename(folder)
    out_path = os.path.join(folder, "prediction_system.txt")
    
    if not os.path.exists(out_path):
        continue
    
    system_gt, _, _ = uu.gen_testdata(conf, matlab_dir)
    data = np.loadtxt(out_path)

    x, t, sys = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:3].T
    X = np.vstack((x, t)).T
    y_sys = sys.flatten()[:, None]
    pred = {"grid": X, "theta": y_sys.reshape(system_gt["theta"].shape), "label": "pred_inv"}

    MSE = uu.calculate_mse(system_gt["theta"], pred["theta"])
    print(f"{run_name}: {MSE.round(10)}")
