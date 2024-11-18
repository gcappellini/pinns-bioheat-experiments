import os
import subprocess
from omegaconf import OmegaConf
import utils as uu
import common as co
import plots as pp
import numpy as np
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

conf_dir = os.path.join(src_dir, "configs")
# cfg = OmegaConf.load(f"{conf_dir}/config_run.yaml")
cfg = OmegaConf.load(f"{src_dir}/config.yaml")

ks = [0.001207]

for WW in ks:
    cfg.model_parameters.W_sys = WW
    cfg.model_properties.W = WW
    cfg.model_parameters.W_obs = WW
    OmegaConf.save(cfg, f"{src_dir}/config.yaml")
    subprocess.run(["python", f'{src_dir}/main.py'])

