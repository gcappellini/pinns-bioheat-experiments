import utils as uu
import os
import numpy as np
from omegaconf import OmegaConf

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)

conf = OmegaConf.load(f"{src_dir}/config.yaml")
s = uu.get_plot_params(conf)
print(s)
