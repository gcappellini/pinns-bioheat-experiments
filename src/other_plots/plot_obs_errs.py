import deepxde as dde
import numpy as np
import os
import torch
import glob
import json
import hashlib
import logging
from omegaconf import OmegaConf
from hydra import compose
import plots as pp
import matplotlib.pyplot as plt
from utils import rescale_t

# device = torch.device("cpu")
device = torch.device("cuda")

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
conf_dir = os.path.join(src_dir, "configs")
os.makedirs(tests_dir, exist_ok=True)

output_folder = f"{tests_dir}/errs_meas"
cfg = OmegaConf.load(f"{conf_dir}/config_run.yaml")


a = dict(np.load(os.path.join(output_folder, "err_meas.npz")))
# b = dict(np.load(os.path.join(output_folder, "l2_errs_meas.npz")))

xref = 0.0
# l2_cool_1 = {k.replace('meas_cool_1_', '').replace('_l2', ''): v for k, v in sorted(a.items()) if k.startswith('meas_cool_1') and k.endswith('l2')}
l2_cool_2 = {k.replace('meas_cool_2_', '').replace('_l2', ''): v for k, v in sorted(a.items()) if k.startswith('meas_cool_2') and k.endswith('l2')}
# y2_cool_1 = {k.replace('meas_cool_1_', '').replace('_0.0', ''): v for k, v in sorted(a.items()) if k.startswith('meas_cool_1') and k.endswith('0.0')}
# y2_cool_2 = {k.replace('meas_cool_2_', '').replace('_0.0', ''): v for k, v in sorted(a.items()) if k.startswith('meas_cool_2') and k.endswith('0.0')}
# gt_cool_2 = {k.replace('meas_cool_1_', '').replace('_0.14', ''): v for k, v in sorted(a.items()) if k.startswith('meas_cool_1') and k.endswith('0.14')}
# gt_cool_2 = {k.replace('meas_cool_2_', '').replace('_0.14', ''): v for k, v in sorted(a.items()) if k.startswith('meas_cool_2') and k.endswith('0.14')}


l2_cool_2 = dict(sorted(l2_cool_2.items(), key=lambda item: int(item[0])))
length = len(l2_cool_2["8"])

times = np.array([np.linspace(0, 1, num=length)]*3)*1800
vals = [ v for v in l2_cool_2.values()]
legend_labels = [fr"$n_{{\mathrm{{obs}}}}={k}$" for k in l2_cool_2.keys()]
y_plot = np.array(rescale_t(vals, conf=cfg)) -21.5

pp.plot_generic(
    x=times,
    y=vals,
    title="Prediction error norm",
    xlabel=r"$t \, (s)$",
    ylabel=r"$L^2$ norm",
    legend_labels=legend_labels,
    log_scale=True,  # We want a log scale on the y-axis
    filename=f"{output_folder}/l2_cooling_2_rsc_bis.png",
    size=(6, 5),
    colors=["#1f77b4", "#2ca02c", "#d62728"],  # Different shades of blue, green, and red
    sma=True
    )


# pp.plot_generic(
#     x=times,
#     y=y_plot,
#     title=f"{round(0.07*xref*100,0)} cm depth",
#     ylabel=r"Error $^{\circ} C$",
#     xlabel=r"$t \, (s)$",
#     legend_labels=legend_labels,
#     filename=f"{output_folder}/y2_cooling_2_bis.png",
#     size=(6, 5),
#     colors=["#1f77b4", "#2ca02c", "#d62728"],  # Different shades of blue, green, and red
#     sma=True
#     )
