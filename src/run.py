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
cfg = OmegaConf.load(f"{conf_dir}/config_run.yaml")

ks = [1, 2, 10, 50, 100, 200, 250, 500]

for K in ks:
    cfg.experiment.name = ["cooling", f"simulation_K{K}"]
    cfg.experiment.type.cooling.simulation.K = K
    OmegaConf.save(cfg, f"{src_dir}/config.yaml")
    subprocess.run(["python", f'{src_dir}/main.py'])

