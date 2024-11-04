import utils as uu
import os
import numpy as np
from omegaconf import OmegaConf

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)


models = os.path.join(git_dir, "models")
os.makedirs(models, exist_ok=True)

a = OmegaConf.load(f"{src_dir}/config.yaml")
x, truths, _, mm_truths = uu.gen_testdata(a) 
g = np.hstack((x, truths))
grid = uu.gen_obsdata(a)

y2 = grid[:, 1]
sys_y2 = g[g[:, 0]==0][:, -1]
print(y2 - sys_y2)
