import utils as uu
import os
import coeff_calc as cc
import plots as pp
import numpy as np
from omegaconf import OmegaConf
import subprocess
import common as co

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)


conf = OmegaConf.load(f"{src_dir}/config.yaml")
set = conf.experiment.name
rescale = conf.plot.rescale
n_obs = conf.model_parameters.n_obs

subprocess.run(["python", f'{src_dir}/ground_truth.py'])

e = uu.import_testdata(f"{set[0]}_{set[1]}")
measurements_tf = e[e[:, 1]==e[:, 1].max()][:,2]
x_measurements = np.unique(e[:, 0])

f = np.hstack(uu.gen_testdata(n_obs))

matlab_tf = f[f[:, 1]==f[:, 1].max()][:,2:]

x_matlab = np.unique(f[:, 0])
x_matlab = x_matlab.reshape(len(x_matlab), 1)

x = [x_measurements] + [x_matlab for _ in range(n_obs + 1)]
y = [measurements_tf]+ [matlab_tf[:, j] for j in range(n_obs+1)]
tauf = conf.model_properties.tauf
title = f"Comparison at t={tauf} s" if rescale else r"Comparison at $\tau=1$"

xlabel, _, ylabel = uu.get_scaled_labels(rescale)

fname = f"{src_dir}/data/vessel/tf_{set[0]}_{set[1]}_{n_obs}obs.png"
obs_colors = uu.get_obs_colors(conf)
true_color, mm_obs_color = uu.get_sys_mm_colors(conf)
obs_linestyles = uu.get_obs_linestyles(conf)
true_linestyle, mm_obs_linestyle = uu.get_sys_mm_linestyle(conf)
meas_color = conf.plot.colors.measuring_points
meas_ls = conf.plot.linestyles.measuring_points

legend_labels = ["Measurements"] + ['System'] + [f'Obs {i}' for i in range(n_obs)] + ['MultiObs']
colors = [meas_color[0]]+[true_color] + obs_colors + [mm_obs_color]
linestyles = [meas_ls[0]]+[true_linestyle] + obs_linestyles + [mm_obs_linestyle]
x_plot = uu.rescale_x(x) if rescale else x
y_plot = uu.rescale_t(y) if rescale else y

pp.plot_generic(x=x_plot,
                y=y_plot,
                title=title,
                xlabel = xlabel,
                ylabel=ylabel,
                legend_labels=legend_labels,
                filename=fname,
                colors=colors,
                linestyles=linestyles)

