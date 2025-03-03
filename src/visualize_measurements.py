import os
import numpy as np
from omegaconf import OmegaConf
import hydra
# import common as co
import plots as pp
# import v1v2_calc as cc
from scipy import integrate
import utils as uu
import time
import torch.nn as nn
import deepxde as dde
import torch
from common import set_run, generate_config_hash
# import matlab.engine
import utils as uu 
import common as co
from matplotlib import colors as mcolors

np.random.seed(237)

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
conf_dir = os.path.join(src_dir, "configs")
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
models = os.path.join(git_dir, "models")
os.makedirs(tests_dir, exist_ok=True)


config = OmegaConf.load(f"{conf_dir}/config_run.yaml")


# print(timeseries_data.keys())
exp_strs = ["meas_cool_1", "meas_cool_2"]

for exp_str in exp_strs:
    meas = getattr(config.experiment_type, exp_str)
    file_path = f"{src_dir}/data/vessel/{meas.date}.txt"
    timeseries_data = uu.load_measurements(file_path)

    keys_to_neglect = [11, 13, 47, 65, 26]
    default_tc = {10:'y1', 45:'gt1', 66:'gt', 24:'y2'}

    for pt_nmb, pt_lbl in default_tc.items():

        keys_to_extract = {pt_nmb: pt_lbl}
        for offset in range(-3, 4):
            if offset != 0:
                key = pt_nmb + offset
                if key not in keys_to_neglect:
                    keys_to_extract[key] = f'{pt_lbl}{offset:+d}'
        dictionary = uu.extract_entries(timeseries_data, meas.start_min*60, meas.end_min*60, keys_to_extract=keys_to_extract)
        # print(dictionary.columns.tolist())
        labels = dictionary.columns.tolist()[1:]

        x = [(dictionary['t']-dictionary['t'][0])/60]*len(labels)
        y = [dictionary[lal] for lal in labels]
        plot_params = uu.get_plot_params(config)
        def generate_similar_colors(base_color, num_colors):
            base_rgb = mcolors.to_rgb(base_color)
            similar_colors = [base_rgb]
            for i in range(1, num_colors):
                similar_color = tuple(min(1, max(0, c + np.random.uniform(-0.1, 0.1))) for c in base_rgb)
                similar_colors.append(similar_color)
            return similar_colors

        base_color = plot_params[pt_lbl]['color']
        colors = generate_similar_colors(base_color, len(labels))
        markers = [''] + ['o', 's', 'D', '+', 'x', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'd', '|', '_'][:len(labels)-1]
        markersize = [1] * len(markers)
        markersize = [7] * len(markers)

        pp.plot_generic(x=x, y=y, title=f"Test {pt_lbl} {meas.title}", xlabel="Time (min)", ylabel="Temperature (°C)", 
                        filename=f"{tests_dir}/{exp_str}/visualize_{exp_str}_{pt_lbl}.png", legend_labels=labels, 
                        colors=colors, markers=markers, markersizes=markersize, markevery=20)


