# import deepxde as dde
import numpy as np
import os
import torch
# import glob
# import json
# import hashlib
# import logging
# from omegaconf import OmegaConf
# from hydra import compose
# import plots as pp
import matplotlib.pyplot as plt
import csv

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

csv_folder = f"{tests_dir}/wandb_results"
output_folder = f"{tests_dir}/wandb_results"

# Load the opt_ai_nbhs.csv file
csv_file = os.path.join(csv_folder, "opt_ai_nbhs.csv")
ai_nbhs = {}

with open(csv_file, mode='r') as file:
    reader = csv.reader(file)
    headers = next(reader)  # First row as keys
    for header in headers:
        ai_nbhs[header] = []
    for row in reader:
        for header, value in zip(headers, row):
            ai_nbhs[header].append(value)

            ai_nbhs["a2wb"] = [float(a2) * float(wb) for a2, wb in zip(ai_nbhs["a2"], ai_nbhs["wb"])]


    ai_nbhs.pop("a2", None)
    ai_nbhs.pop("wb", None)
    ai_nbhs = {k: ai_nbhs[k] for k in sorted(ai_nbhs.keys())}
# print(ai_nbhs)
# print("Keys in ai_nbhs:", ai_nbhs.keys())

# Convert data to numpy arrays
a1 = np.array(ai_nbhs["a1"], dtype=float)
a2wb = np.array(ai_nbhs["a2wb"], dtype=float)
mse = np.array(ai_nbhs["mse"], dtype=float)
testloss = np.array(ai_nbhs["testloss"], dtype=float)

# Create scatter plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(a1, a2wb, s=280, c=mse, vmax= 10*mse.min(), cmap='viridis_r', edgecolor='k')
cbar = plt.colorbar(scatter)
cbar.set_label("MSE", fontsize=legend_fs, fontweight="bold")
plt.xlabel(r"$\mathbf{a_2 w_b}$", fontweight="bold", fontsize=axis_fs)
plt.ylabel(r"$\mathbf{a_1}$", fontweight="bold", fontsize=axis_fs)
plt.title("MSE", fontweight="bold", fontsize=title_fs)
cbar.ax.tick_params(labelsize=tick_fs) 
cbar.ax.yaxis.get_offset_text().set_fontsize(tick_fs)
plt.tick_params(axis='both', which='both', labelsize=tick_fs) 
plt.grid(True)
plt.tight_layout()

# Save the plot
output_path = os.path.join(output_folder, "mse_vs_a1_a2wb.png")
plt.savefig(output_path)
# plt.show()

# Create scatter plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(a1, a2wb, s=280, c=testloss, vmax=20*testloss.min(), cmap='inferno_r', edgecolor='k')
cbar = plt.colorbar(scatter)
cbar.set_label("Test Loss", fontsize=legend_fs, fontweight="bold")
plt.xlabel(r"$\mathbf{a_2 w_b}$", fontweight="bold", fontsize=axis_fs)
plt.ylabel(r"$\mathbf{a_1}$", fontweight="bold", fontsize=axis_fs)
plt.title("Test Loss", fontweight="bold", fontsize=title_fs)
cbar.ax.tick_params(labelsize=tick_fs) 
plt.tick_params(axis='both', which='both', labelsize=tick_fs) 
plt.grid(True)
plt.tight_layout()

# Save the plot
output_path = os.path.join(output_folder, "testloss_vs_a1_a2wb.png")
plt.savefig(output_path)
plt.show()

# Save all arrays into a single .npz file
# a = np.load(os.path.join(output_folder, "l2_errs_meas.npz"))

# l2_cool_1 = {k.replace('meas_cool_1_', '').replace('_obs', ''): v for k, v in sorted(a.items()) if k.startswith('meas_cool_1')}
# l2_cool_2 = {k.replace('meas_cool_2_', '').replace('_obs', ''): v for k, v in sorted(a.items()) if k.startswith('meas_cool_2')}

# length = len(l2_cool_1["8"])

# times = np.array([np.linspace(0, 1, num=length)]*3)
# vals = [ v for v in l2_cool_1.values()]
# legend_labels = [fr"$n_{{\mathrm{{obs}}}}={k}$" for k in l2_cool_1.keys()]

# pp.plot_generic(
#         x=times,
#         y=vals,
#         title="Prediction error norm",
#         xlabel=r"$\tau$",
#         ylabel=r"$L^2$ norm",
#         legend_labels=legend_labels,
#         log_scale=True,  # We want a log scale on the y-axis
#         filename=f"{output_folder}/ai_nbhs_testloss.png",
#         size=(6, 5),
#         scatter=True
#         # colors=colors
#     )

