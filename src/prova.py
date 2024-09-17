import numpy as np
import os
import plots as pp
import common as co
import utils as uu

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

prj = co.set_prj("change_matlab")
run = co.set_run("perfusion_muscle")

X, y_sys, y_obs, y_mmobs = uu.gen_testdata()
Xobs_y1, y1 = uu.gen_obs_y1()
Xobs_y2, y2 = uu.gen_obs_y2()

tot = np.hstack((X, y_sys, y_obs, y_mmobs))
rows_x0 = tot[tot[:,0]==0]
rows_x1 = tot[tot[:,0]==1]

y = np.vstack([y2, rows_x0[:, 4:5].reshape(y2.shape), y1, rows_x1[:, 4:5].reshape(y2.shape)])
legend_labels = [r'$y_2(\tau)$', r'$\hat{\theta}(0, \tau)$', r'$y_1(\tau)$', r'$\hat{\theta}(1, \tau)$']
pp.plot_generic(Xobs_y1[:, -1], y, "Matlab comparison at the boundary", r"Time ($\tau$)", r"Theta ($\theta$)", legend_labels, filename=f"{run}/comparison_outputs_matlab_W1")