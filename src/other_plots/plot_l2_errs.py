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


# device = torch.device("cpu")
device = torch.device("cuda")

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
conf_dir = os.path.join(src_dir, "configs")
os.makedirs(tests_dir, exist_ok=True)

output_folder = f"{tests_dir}/l2_errs_meas"


# Save all arrays into a single .npz file
a = np.load(os.path.join(output_folder, "l2_errs_meas.npz"))

l2_cool_1 = {k.replace('meas_cool_1_', '').replace('_obs', ''): v for k, v in sorted(a.items()) if k.startswith('meas_cool_1')}
l2_cool_2 = {k.replace('meas_cool_2_', '').replace('_obs', ''): v for k, v in sorted(a.items()) if k.startswith('meas_cool_2')}

length = len(l2_cool_1["8"])

times = np.array([np.linspace(0, 1, num=length)]*3)
vals = [ v for v in l2_cool_1.values()]
legend_labels = [fr"$n_{{\mathrm{{obs}}}}={k}$" for k in l2_cool_1.keys()]

pp.plot_generic(
        x=times,
        y=vals,
        title="Prediction error norm",
        xlabel=r"$\tau$",
        ylabel=r"$L^2$ norm",
        legend_labels=legend_labels,
        log_scale=True,  # We want a log scale on the y-axis
        filename=f"{output_folder}/cooling_1.png",
        size=(6, 5),
        # colors=colors
    )

