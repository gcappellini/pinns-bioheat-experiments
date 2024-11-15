from utils import scale_t, ic_obs, run_matlab_ground_truth
import os
import numpy as np
from omegaconf import OmegaConf
import hydra
np.random.seed(237)

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)

print(0.14*np.exp(-1))

