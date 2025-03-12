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

# Initialize dictionaries
meas_cool_1 = {}
meas_cool_2 = {}

# Get all txt files in the directory
txt_files = glob.glob(os.path.join(output_folder, "*.txt"))

# Read each file and store its content in the appropriate dictionary
for file_path in txt_files:
    file_name = os.path.basename(file_path)  # Extract only the filename
    file_key = os.path.splitext(file_name)[0]  # Remove .txt extension
    
    parts = file_key.split("_")  # Split by "_"
    prefix = f"{parts[0]}_{parts[1]}_{parts[2]}"  # e.g., "meas_cool_1" or "meas_cool_2"
    obs_key = f"{parts[3]}"  # e.g., "8obs", "16obs", "64obs"

    with open(file_path, "r") as file:
        value = file.read().strip()  # Read file content

    # Store in the corresponding dictionary
    if prefix == "meas_cool_1":
        meas_cool_1[obs_key] = np.array(value)
    elif prefix == "meas_cool_2":
        meas_cool_2[obs_key] = np.array(value)

# Print the results
length = meas_cool_1["8"]
print(length)
# print("meas_cool_2:", meas_cool_2.keys())


# times = np.array([np.linspace(0, 1, num=len(meas_cool_1["8"]))]*3)
# vals = [ v for v in meas_cool_1.values()]
# legend_labels = [f"{k}" for k in meas_cool_1.keys()]

# pp.plot_generic(
#         x=times,
#         y=vals,
#         title="Prediction error norm",
#         xlabel=r"$\tau$",
#         ylabel=r"$L^2$ norm",
#         legend_labels=legend_labels,
#         log_scale=True,  # We want a log scale on the y-axis
#         filename=f"{output_folder}/cooling_1.png",
#         size=(6, 5),
#         # colors=colors
#     )