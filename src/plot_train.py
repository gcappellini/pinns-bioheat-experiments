# import deepxde as dde
import numpy as np
import os
import torch
# import glob
# import json
# import hashlib
# import logging
# from omegaconf import OmegaConf
from hydra import compose, initialize
import plots as pp
import matplotlib.pyplot as plt
import csv
import common as co
import utils as uu

title_fs = 26
axis_fs = 20
axis3d_fs = 12
tick_fs = 20
legend_fs = 20

# device = torch.device("cpu")
device = torch.device("cuda")

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
conf_dir = os.path.join(src_dir, "configs")
os.makedirs(tests_dir, exist_ok=True)

pinns="nbho"

output_folder = f"{tests_dir}/plot_train_{pinns}"
os.makedirs(output_folder, exist_ok=True)

# conf
initialize('./configs', version_base=None) 
config = compose(config_name='config_run')
label = "simulation_mm_obs" if pinns == "nbho" else "simulation_system"
hps, exp = config.hp, config.experiment

# widths = [10, 20, 40, 80, 160] 
# depths = [1, 2, 3, 4, 5, 6]
activations = ["tanh", "relu", "sin", "swish", "sigmoid"]
# res = {}

# for depth in depths:
#     hps.depth = depth
#     hps.width = 50
#     sub_output_folder = f"{output_folder}/D_{hps.depth}_W_{hps.width}"
#     os.makedirs(sub_output_folder, exist_ok=True)
#     config.output_dir = sub_output_folder
#     output_dir_system, config = co.set_run(sub_output_folder, config, label)

#     _, losshistory = uu.train_model(config)
#     steps = losshistory["steps"]
#     train_loss = losshistory["train"].sum(axis=1)
#     mask = steps <= 7000
#     res[f"{depth}D - 50"] = np.array([losshistory["steps"], losshistory["train"].sum(axis=1)])

# np.savez(os.path.join(output_folder, f"res_{pinns}_D.npz"), **res)

# res = {}

# for width in widths:
#     hps.depth = 4
#     hps.width = width
#     sub_output_folder = f"{output_folder}/D_{hps.depth}_W_{hps.width}"
#     os.makedirs(sub_output_folder, exist_ok=True)
#     config.output_dir = sub_output_folder
#     output_dir_system, config = co.set_run(sub_output_folder, config, label)

#     _, losshistory = uu.train_model(config)
#     steps = losshistory["steps"]
#     train_loss = losshistory["train"].sum(axis=1)
#     mask = steps <= 7000
#     res[f"4 - {hps.width}W"] = np.array([steps[mask], train_loss[mask]])

# np.savez(os.path.join(output_folder, f"res_{pinns}_W.npz"), **res)

res = {}

for act in activations:
    hps.depth = 4
    hps.width = 50
    hps.af = act
    sub_output_folder = f"{output_folder}/A_{hps.af}"
    os.makedirs(sub_output_folder, exist_ok=True)
    config.output_dir = sub_output_folder
    output_dir_system, config = co.set_run(sub_output_folder, config, label)

    _, losshistory = uu.train_model(config)
    steps = losshistory["steps"]
    train_loss = losshistory["train"].sum(axis=1)
    mask = steps <= 7000
    res[f"{hps.af}"] = np.array([steps[mask], train_loss[mask]])

np.savez(os.path.join(output_folder, f"res_{pinns}_A.npz"), **res)

# a = np.load(os.path.join(output_folder, f"res_{pinns}_W.npz"))

# vals = [ v[1] for v in a.values()]
# epochs = [ v[0] for v in a.values()]

# legend_labels = [f"$\\mathrm{{{key}}}$" for key in a.keys()]
# colors = ['blue'] * len(legend_labels) if pinns=="nbho" else ['black'] * len(legend_labels)
# markers = ['s', '1', 'o', '3', '*', 'h', 'x', '^', 'v', '+'][:len(legend_labels)]
# markersizes = [5] * len(legend_labels)
# linewidths = [0.2] * len(legend_labels)

pinns_title = "NBHO" if pinns == "nbho" else "NBHS"

# pp.plot_generic(
#     x=epochs,
#     y=vals,
#     title=f"{pinns_title} - Varying Width",
#     xlabel="Epoch",
#     ylabel="Training Error",
#     legend_labels=legend_labels,
#     log_scale=True,  # We want a log scale on the y-axis
#     filename=f"{output_folder}/{pinns}_width.png",
#     size=(6, 5),
#     colors=colors,
#     markers=markers,
#     markersizes=markersizes,
#     linewidths=linewidths,)

# b = np.load(os.path.join(output_folder, f"res_{pinns}_D.npz"))

# vals = [ v[1] for v in b.values()]
# epochs = [ v[0] for v in b.values()]

# legend_labels = list(b.keys())
# colors = ['blue'] * len(legend_labels) if pinns=="nbho" else ['black'] * len(legend_labels)
# markers = ['v', '2', 'o', '+', 'p', 'x', '*', 'h', 'D', '+'][:len(legend_labels)]
# markersizes = [5] * len(legend_labels)
# linewidths = [0.2] * len(legend_labels)


# pp.plot_generic(
#     x=epochs,
#     y=vals,
#     title=f"{pinns_title} - Varying Depth",
#     xlabel="Epoch",
#     ylabel="Training Error",
#     legend_labels=legend_labels,
#     log_scale=True,  # We want a log scale on the y-axis
#     filename=f"{output_folder}/{pinns}_depth.png",
#     size=(6, 5),
#     colors=colors,
#     markers=markers,
#     markersizes=markersizes,
#     linewidths=linewidths,)

c = np.load(os.path.join(output_folder, f"res_{pinns}_A.npz"))

vals = [ v[1] for v in c.values()]
epochs = [ v[0] for v in c.values()]

legend_labels = list(c.keys())
colors = ['blue'] * len(legend_labels) if pinns=="nbho" else ['black'] * len(legend_labels)
markers = ['o', '3', '', '+', 'D', 'x', '*', 'h', 'D', '+'][:len(legend_labels)]
markersizes = [5] * len(legend_labels)
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--'][:len(legend_labels)]
linewidths = [0.2] * len(legend_labels)
linewidths[2] = 1.5


pp.plot_generic(
    x=epochs,
    y=vals,
    title=f"{pinns_title} - Activation",
    xlabel="Epoch",
    ylabel="Training Error",
    legend_labels=legend_labels,
    log_scale=True,  # We want a log scale on the y-axis
    filename=f"{output_folder}/{pinns}_af.png",
    size=(6, 5),
    colors=colors,
    markers=markers,
    markersizes=markersizes,
    linewidths=linewidths,
    linestyles=linestyles)

