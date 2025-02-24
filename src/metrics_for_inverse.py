import pandas as pd 
import wandb
import os
import yaml
import csv
import utils as uu
from omegaconf import OmegaConf
import numpy as np

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
conf_dir = os.path.join(src_dir, "configs")
models = os.path.join(git_dir, "models")
tests_dir = os.path.join(git_dir, "tests")

run_name = "inverse_2x100"
matlab_dir = f"{tests_dir}/cooling_ground_truth_5e-04"
out_dir = f"{tests_dir}/cooling_inverse/{run_name}"

conf = OmegaConf.load(f"{conf_dir}/config_run.yaml")
props = conf.model_properties
pars = conf.model_parameters

system_gt, _, _ = uu.gen_testdata(conf, matlab_dir)
data = np.loadtxt(f"{out_dir}/prediciton_system.txt")

x, t, sys = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:3].T
X = np.vstack((x, t)).T
y_sys = sys.flatten()[:, None]
pred = {"grid": X, "theta": y_sys, "label": "pred_inv"}

metrics = uu.compute_metrics([system_gt, pred], {"run": run_name}, conf, out_dir)
