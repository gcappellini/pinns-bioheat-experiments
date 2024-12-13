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
from common import set_run, generate_config_hash
from simulation import load_ground_truth

np.random.seed(237)

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
conf_dir = os.path.join(src_dir, "configs")
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
models = os.path.join(git_dir, "models")
os.makedirs(tests_dir, exist_ok=True)

fold = f"{tests_dir}/transfer_learning"
os.makedirs(fold, exist_ok=True)

# Step 0: load model with n_ins=2

conf = OmegaConf.load(f"{conf_dir}/config_run.yaml")

output_dir_gt, system_gt, observers_gt, mm_obs_gt = load_ground_truth(conf, f"{tests_dir}/cooling_simulation")

l2_err = uu.calculate_l2(system_gt["grid"], system_gt["theta"], mm_obs_gt["theta"])
print(l2_err.shape, system_gt["theta"].shape)
print(l2_err)