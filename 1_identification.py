import numpy as np
import os
import matplotlib.pyplot as plt


current_file = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file)

output_dir = f"{script_directory}/identification"
meas_dir = f"{script_directory}/measurements"

os.makedirs(output_dir, exist_ok=True)

labels = [["AX1", "AX2", "BX1", "BX2"], ["AY1", "AY2", "BY1", "BY2"]]
k = 0.563
L_0 = 0.15
lab = "h"
# lab = "fl"
# lab = "y3y2"

# Create figure 5
fig5, axs5 = plt.subplots(2, 2, figsize=(13, 7))

# Load and plot data for figure 5
for i, label in enumerate(labels[0]):
    # Load the saved data
    obs = np.load(f"{meas_dir}/obs_{label}.npz")
    t, y_1, y_2, y_3 = obs["t"], obs["t_0"], obs["t_1"], obs["t_bolus"] 

    pl = (y_2 - y_1)*k/((y_3 - y_2)*L_0)
    # pl = (y_2 - y_1)*k/L_0
    # pl = y_3 - y_2

    axs5[i//2, i%2].plot(t, pl, 'm-', label=f"{lab}")
    axs5[i//2, i%2].set_xlabel(r"$\tau$", fontsize=12)
    axs5[i//2, i%2].set_ylabel(f"{lab}", fontsize=12)
    axs5[i//2, i%2].set_title(f"{label}", fontsize=14, fontweight="bold")
    axs5[i//2, i%2].tick_params(axis='both', which='major', labelsize=10)

# Adjust layout
plt.tight_layout()
plt.savefig(f'{output_dir}/{lab}_X.png')
plt.show()
plt.close()
plt.clf()


# Create figure 6
fig6, axs6 = plt.subplots(2, 2, figsize=(13, 7))

# Load and plot data for figure 4
for i, label in enumerate(labels[1]):
    # Load the saved data
    obs = np.load(f"{meas_dir}/obs_{label}.npz")
    t, y_1, y_2, y_3 = obs["t"], obs["t_0"], obs["t_1"], obs["t_bolus"] 

    pl = (y_2 - y_1)*k/((y_3 - y_2)*L_0)
    # pl = (y_2 - y_1)*k/L_0
    # pl = y_3 - y_2

    axs6[i//2, i%2].plot(t, pl, 'm-', label=f"{lab}")
    axs6[i//2, i%2].set_xlabel(r"$\tau$", fontsize=12)
    axs6[i//2, i%2].set_ylabel(f"{lab}", fontsize=12)
    axs6[i//2, i%2].set_title(f"{label}", fontsize=14, fontweight="bold")
    axs6[i//2, i%2].tick_params(axis='both', which='major', labelsize=10)

# Adjust layout
plt.tight_layout()
plt.savefig(f'{output_dir}/{lab}_Y.png')
plt.show()
plt.close()
plt.clf()


good_labels = ["AY1", "AY2", "BY1", "BX2"]

# Create figure 7
fig7, axs7 = plt.subplots(2, 2, figsize=(13, 7))

# Load and plot data for figure 4
for i, label in enumerate(good_labels):
    # Load the saved data
    obs = np.load(f"{meas_dir}/obs_{label}.npz")
    t, y_1, y_2, y_3 = obs["t"], obs["t_0"], obs["t_1"], obs["t_bolus"] 

    pl = (y_2 - y_1)*k/((y_3 - y_2)*L_0)
    # pl = (y_2 - y_1)*k/L_0
    # pl = y_3 - y_2

    # Only consider values where t > 0.6
    t_06 = t[t > 0.6]
    pl_06 = pl[t > 0.6]

    axs7[i//2, i%2].plot(t_06, pl_06, 'm-', label=f"{lab}")
    axs7[i//2, i%2].set_xlabel(r"$\tau$", fontsize=12)
    axs7[i//2, i%2].set_ylabel(f"{lab}", fontsize=12)
    axs7[i//2, i%2].set_title(f"{label}", fontsize=14, fontweight="bold")
    axs7[i//2, i%2].tick_params(axis='both', which='major', labelsize=10)

# Adjust layout
plt.tight_layout()
plt.savefig(f'{output_dir}/{lab}_good.png')
plt.show()
plt.close()