import os
import subprocess
from omegaconf import OmegaConf
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)



cfg = OmegaConf.load(f"{src_dir}/config.yaml")

exps = ["meas_1", "meas_2"]

for exp in range(len(exps)):
    cfg.experiment.name = ["cooling", exps[exp]]
    OmegaConf.save(cfg, f"{src_dir}/config.yaml")
    subprocess.run(["python", f'{src_dir}/main.py'])
    subprocess.run(["python", f'{src_dir}/ic_compatibility_conditions.py'])