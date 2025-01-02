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
from simulation import load_ground_truth
import matlab.engine
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

fold = f"{tests_dir}/cooling_interp"
config = OmegaConf.load(f"{conf_dir}/config_run.yaml")
os.makedirs(fold, exist_ok=True)

eng = matlab.engine.start_matlab()
eng.cd(f"{src_dir}/matlab", nargout=0)
eng.BioHeat(nargout=0)
eng.quit()

label = "ground_truth"
output_dir_gt, config_matlab = co.set_run(fold, config, label)

config_matlab = OmegaConf.load(f"{conf_dir}/config_ground_truth.yaml")
output_dir_gt = f"{fold}/ground_truth"

system_gt, observers_gt, mm_obs_gt = uu.gen_testdata(config_matlab, path=output_dir_gt)
uu.check_and_wandb_upload(
    label=label,
    mm_obs_gt=mm_obs_gt,
    system_gt=system_gt,
    conf=config,
    output_dir=output_dir_gt,
    observers_gt=observers_gt
)