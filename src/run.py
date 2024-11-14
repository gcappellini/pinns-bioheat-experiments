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

ks = [(6.3, 0.9), (4.3762, 0.4256), (4.0865, 0.0), (4.6887, 0.2422)]

for (b1, b2) in ks:
    cfg.model_properties.b1 = b1
    cfg.model_properties.b2 = b2
    OmegaConf.save(cfg, f"{src_dir}/config.yaml")
    subprocess.run(["python", f'{src_dir}/main.py'])

