import os
import subprocess
from omegaconf import OmegaConf
import utils as uu
import common as co
import plots as pp
import numpy as np
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)



# cfg = OmegaConf.load(f"{src_dir}/config.yaml")

# exps = ["meas_1", "meas_2"]

# for exp in range(len(exps)):
#     cfg.experiment.name = ["cooling", exps[exp]]
#     OmegaConf.save(cfg, f"{src_dir}/config.yaml")
#     subprocess.run(["python", f'{src_dir}/main.py'])
#     subprocess.run(["python", f'{src_dir}/ic_compatibility_conditions.py'])
conf = OmegaConf.load(f"{src_dir}/config.yaml")
output = co.set_prj("compare_single_obs")
uu.mm_observer(conf)

data = np.loadtxt(f"{output}/output_matlab_1Obs.txt")
x, t, sys, y_obs = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:3].T, data[:, 3:4].T   

obs = y_obs.flatten()[:, None]

X = np.vstack((x, t)).T
y_sys = sys.flatten()[:, None]

pp.plot_tf_matlab_1obs(X, y_sys, obs, output)
pp.plot_l2_matlab_1obs(X, y_sys, obs, output)
pp.plot_comparison_3d(X, y_sys, obs, output, gt= True)