import os
import subprocess
from omegaconf import OmegaConf
import utils as uu
import common as co
import plots as pp
import numpy as np
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

cfg = OmegaConf.load(f"{src_dir}/config.yaml")
cfg.model_parameters.lam = 150
cfg.model_parameters.upsilon = 5
cfg.experiment.run_matlab = True

# n_obs = [1, 8]

# for n in n_obs:
#     cfg.experiment.name = ["cooling", "simulation"]
#     cfg.model_parameters.n_obs = n
#     OmegaConf.save(cfg, f"{src_dir}/config.yaml")
#     OmegaConf.save(cfg, f"{src_dir}/configs/cooling_simulation_{n}obs.yaml")
#     subprocess.run(["python", f'{src_dir}/main.py'])

# cfg.model_parameters.n_obs = 8
# exps = ["meas_1", "meas_2"]

# for exp in range(len(exps)):
#     cfg.experiment.name = ["cooling", exps[exp]]
#     OmegaConf.save(cfg, f"{src_dir}/config.yaml")
#     OmegaConf.save(cfg, f"{src_dir}/configs/cooling_{exps[exp]}.yaml")
#     subprocess.run(["python", f'{src_dir}/main.py'])

cfg.model_parameters.lam = 750
cfg.model_parameters.upsilon = 50
cfg.experiment.run_matlab = False

# for n in n_obs:
#     cfg.experiment.name = ["cooling", "simulation"]
#     cfg.model_parameters.n_obs = n
#     OmegaConf.save(cfg, f"{src_dir}/config.yaml")
#     OmegaConf.save(cfg, f"{src_dir}/configs/cooling_simulation_{n}obs_enhanced.yaml")
#     subprocess.run(["python", f'{src_dir}/main.py'])

cfg.model_parameters.n_obs = 8
exps = ["meas_1", "meas_2"]

for exp in range(len(exps)):
    cfg.experiment.name = ["cooling", exps[exp]]
    OmegaConf.save(cfg, f"{src_dir}/config.yaml")
    OmegaConf.save(cfg, f"{src_dir}/configs/cooling_{exps[exp]}_enhanced.yaml")
    subprocess.run(["python", f'{src_dir}/main.py'])