"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import interp1d


current_file = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file)

output_dir = f"{script_directory}/identification"

os.makedirs(output_dir, exist_ok=True)

labels = [["AX1", "AX2", "BX1", "BX2"], ["AY1", "AY2", "BY1", "BY2"]]
k = 0.563
L_0 = 0.15

# Create figure 5
fig5, axs5 = plt.subplots(2, 2, figsize=(13, 7))

# Load and plot data for figure 5
for i, label in enumerate(labels[0]):
    # Load the saved data
    obs = np.load(f"{output_dir}/obs_{label}.npz")
    t, y_1, y_2, y_3 = obs["t"], obs["t_0"], obs["t_1"], obs["t_bolus"] 

    h = (y_2 - y_1)*k/((y_3 - y_2)*L_0)
    # Plot t_0, t_1, and t_bolus against t on each subplot
    axs5[i//2, i%2].plot(t, h, 'm-', label=r'$h$')
    axs5[i//2, i%2].set_xlabel(r"$\tau$", fontsize=12)
    axs5[i//2, i%2].set_ylabel(r"$h$", fontsize=12)
    axs5[i//2, i%2].set_title(f"{label}", fontsize=14, fontweight="bold")
    axs5[i//2, i%2].tick_params(axis='both', which='major', labelsize=10)
    axs5[i//2, i%2].legend()

# Adjust layout
plt.tight_layout()
plt.savefig(f'{output_dir}/h_X.png')
plt.show()
plt.close()


# Create figure 6
fig6, axs6 = plt.subplots(2, 2, figsize=(13, 7))

# Load and plot data for figure 4
for i, label in enumerate(labels[1]):
    # Load the saved data
    obs = np.load(f"{output_dir}/obs_{label}.npz")
    t, y_1, y_2, y_3 = obs["t"], obs["t_0"], obs["t_1"], obs["t_bolus"] 

    h = (y_2 - y_1)*k/((y_3 - y_2)*L_0)
    # Plot observing['t_inf'] vs observing['time'] on the left subplot
    axs5[i//2, i%2].plot(t, h, 'm-', label=r'$h$')
    axs5[i//2, i%2].set_xlabel(r"$\tau$", fontsize=12)
    axs5[i//2, i%2].set_ylabel(r"$h$", fontsize=12)
    axs6[i//2, i%2].set_title(f"{label}", fontsize=14, fontweight="bold")
    axs6[i//2, i%2].tick_params(axis='both', which='major', labelsize=10)
    axs6[i//2, i%2].legend()

# Adjust layout
plt.tight_layout()
plt.savefig(f'{output_dir}/h_Y.png')
plt.show()
plt.close()