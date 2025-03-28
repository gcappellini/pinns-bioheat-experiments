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
# res = {}

# for depth in depths:
#     hps.depth = depth
#     hps.width = 50
#     sub_output_folder = f"{output_folder}/D_{hps.depth}_W_{hps.width}"
#     os.makedirs(sub_output_folder, exist_ok=True)
#     config.output_dir = sub_output_folder
#     output_dir_system, config = co.set_run(sub_output_folder, config, label)

#     _, losshistory = uu.train_model(config)
#     res[f"D_{depth}_W_{hps.width}"] = np.array([losshistory["steps"], losshistory["train"].sum(axis=1)])

# np.savez(os.path.join(output_folder, "res_data_D.npz"), **res)

# res = {}

# for width in widths:
#     hps.depth = 4
#     hps.width = width
#     sub_output_folder = f"{output_folder}/D_{hps.depth}_W_{hps.width}"
#     os.makedirs(sub_output_folder, exist_ok=True)
#     config.output_dir = sub_output_folder
#     output_dir_system, config = co.set_run(sub_output_folder, config, label)

#     _, losshistory = uu.train_model(config)
#     res[f"D_{hps.depth}_W_{hps.width}"] = np.array([losshistory["steps"], losshistory["train"].sum(axis=1)])

# np.savez(os.path.join(output_folder, "res_data_W.npz"), **res)


# a = np.load(os.path.join(output_folder, "res_data_W.npz"))
# filtered_data = {k: v for k, v in a.items() if k.startswith("D_4")}
# np.savez(os.path.join(output_folder, "res_data_W.npz"), **filtered_data)

a = np.load(os.path.join(output_folder, "res_data_W.npz"))


vals = [ v[1] for v in a.values()]
epochs = [ v[0] for v in a.values()]

legend_labels = list(a.keys())
legend_labels = [f"4 - {k.split('_')[-1][2:]}W" for k in legend_labels]
colors = ['blue'] * len(legend_labels) if pinns=="nbho" else ['black'] * len(legend_labels)
markers = ['s', 'd', 'o', 'h', '*', 'p', 'x', '^', 'v', '+'][:len(legend_labels)]
markersizes = [5] * len(legend_labels)
linewidths = [0.2] * len(legend_labels)

pinns_title = "NBHO" if pinns == "nbho" else "NBHS"

pp.plot_generic(
    x=epochs,
    y=vals,
    title=f"{pinns_title} - Varying Width",
    xlabel="Epoch",
    ylabel="Training Error",
    legend_labels=legend_labels,
    log_scale=True,  # We want a log scale on the y-axis
    filename=f"{output_folder}/{pinns}_width.png",
    size=(6, 5),
    colors=colors,
    markers=markers,
    markersizes=markersizes,
    linewidths=linewidths,)

b = np.load(os.path.join(output_folder, "res_data_D.npz"))

vals = [ v[1] for v in b.values()]
epochs = [ v[0] for v in b.values()]

legend_labels = list(b.keys())
legend_labels = [f"{k.split('_')[1][1:]}D - 50" for k in legend_labels]
colors = ['blue'] * len(legend_labels) if pinns=="nbho" else ['black'] * len(legend_labels)
markers = ['v', '2', 'x', '^', 'o', 'p', '*', 'h', 'D', '+'][:len(legend_labels)]
markersizes = [5] * len(legend_labels)
linewidths = [0.2] * len(legend_labels)



pp.plot_generic(
    x=epochs,
    y=vals,
    title=f"{pinns_title} - Varying Depth",
    xlabel="Epoch",
    ylabel="Training Error",
    legend_labels=legend_labels,
    log_scale=True,  # We want a log scale on the y-axis
    filename=f"{output_folder}/{pinns}_depth.png",
    size=(6, 5),
    colors=colors,
    markers=markers,
    markersizes=markersizes,
    linewidths=linewidths,)

