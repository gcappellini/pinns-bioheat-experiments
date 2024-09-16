
# To do: check if gen_obsdata it's correct, by comparinf y2 to theta_o of matlab. 
# implement a function to extract theta 0 and theta 1 from matlab


import utils as uu
import os
import matlab.engine
import numpy as np
from scipy import integrate
import common as co
import plots as pp
import wandb

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

# Ground truth with Matlab:
# eng = matlab.engine.start_matlab()
# eng.cd(src_dir, nargout=0)
# eng.BioHeat(nargout=0)
# eng.quit()

prj = "hpo_240915_bis"
run = "repeat_top"
# PINNs implementation
co.set_prj(prj)
run_figs = co.set_run(run)

config = uu.read_json(f"{src_dir}/properties.json")

aa = {"activation": config["activation"],
        "learning_rate": config["learning_rate"],
        "num_dense_layers": config["num_dense_layers"],
        "num_dense_nodes": config["num_dense_nodes"],
        "initialization": config["initialization"]
        }

wandb.init(
    project=prj, name=run,
    config=aa
)

model = uu.train_model()

X, y_sys, y_obs, _ = uu.gen_testdata()
t = np.unique(X[:, 1])
x_obs = uu.gen_obsdata()
y_pred = model.predict(x_obs)
pp.check_obs(X, y_obs[:, 0], y_pred, 0, run_figs)

errors = uu.compute_metrics(y_obs[:, 0], y_pred)

wandb.log(errors)
wandb.finish()