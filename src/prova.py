import numpy as np
import utils as uu
import os
import subprocess
from omegaconf import OmegaConf
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

subprocess.run(["python", f'{src_dir}/main.py'])

cfg = OmegaConf.load(f"{src_dir}/config.yaml")
cfg.experiment.name = ["cooling", "meas_2"]
OmegaConf.save(cfg, f"{src_dir}/config.yaml")

subprocess.run(["python", f'{src_dir}/main.py'])