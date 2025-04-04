import numpy as np
import os
import matplotlib.pyplot as plt
import csv

# Define font sizes for plots
title_fs = 26
axis_fs = 20
tick_fs = 20
legend_fs = 20

# Define file paths
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)

csv_folder = f"{tests_dir}/wandb_results"
output_folder = f"{tests_dir}/wandb_results"

# Load new CSV file
pinn = "nbho"
csv_file = os.path.join(csv_folder, f"sp_{pinn}.csv")
data = []
headers = []

with open(csv_file, mode='r') as file:
    reader = csv.reader(file)
    headers = next(reader)  # First row as keys
    for row in reader:
        processed_row = []
        for i, value in enumerate(row):
            if i == 2:  # Handle boolean values
                processed_row.append(value.lower() == 'true')
            else:
                processed_row.append(float(value) if i > 2 else int(value))
        data.append(processed_row)

# Convert data to numpy arrays
data = np.array(data, dtype=object)  # Keep boolean values as objects
nres, nb, resampling, runtime, testloss, mse = data.T

# Ensure that resampling is treated as a boolean array
resampling = np.array(resampling, dtype=bool)

# Split the data into resampled and non-resampled
nres_resampled = nres[resampling]
nb_resampled = nb[resampling]
testloss_resampled = testloss[resampling]
mse_resampled = mse[resampling]

nres_non_resampled = nres[~resampling]
nb_non_resampled = nb[~resampling]
testloss_non_resampled = testloss[~resampling]
mse_non_resampled = mse[~resampling]

# Plot for Resampled Data (Test Loss)
plt.figure(figsize=(8, 6))
# Scatter for points with resampling (red circle)
scatter_resampling_testloss = plt.scatter(nres_resampled, nb_resampled, s=280, c=testloss_resampled, 
                                          cmap='inferno_r', edgecolor='r', linewidths=2, marker='o')
cbar = plt.colorbar(scatter_resampling_testloss)
cbar.set_label("Test Loss", fontsize=legend_fs, fontweight="bold")
plt.xlabel(r"$n_{\mathrm{res}}$", fontweight="bold", fontsize=axis_fs)
plt.ylabel(r"$n_{\mathrm{b}}$", fontweight="bold", fontsize=axis_fs)
plt.title("Test Loss - Resampling", fontweight="bold", fontsize=title_fs)
cbar.ax.tick_params(labelsize=tick_fs)
plt.tick_params(axis='both', which='both', labelsize=tick_fs)
plt.grid(True)
plt.tight_layout()

# Save plot for Resampled Test Loss
output_path = os.path.join(output_folder, f"{pinn}_resampled_nres_nb_testloss.png")
plt.savefig(output_path)

# Plot for Resampled Data (MSE)
plt.figure(figsize=(8, 6))
# Scatter for points with resampling (red circle)
scatter_resampling_mse = plt.scatter(nres_resampled, nb_resampled, s=280, c=mse_resampled, 
                                     cmap='viridis_r', edgecolor='r', linewidths=2, marker='o')
cbar = plt.colorbar(scatter_resampling_mse)
cbar.set_label("MSE", fontsize=legend_fs, fontweight="bold")
plt.xlabel(r"$n_{\mathrm{res}}$", fontweight="bold", fontsize=axis_fs)
plt.ylabel(r"$n_{\mathrm{b}}$", fontweight="bold", fontsize=axis_fs)
plt.title("MSE - Resampling", fontweight="bold", fontsize=title_fs)
cbar.ax.tick_params(labelsize=tick_fs)
cbar.ax.yaxis.get_offset_text().set_fontsize(tick_fs)
plt.tick_params(axis='both', which='both', labelsize=tick_fs)
plt.grid(True)
plt.tight_layout()

# Save plot for Resampled MSE
output_path = os.path.join(output_folder, f"{pinn}_resampled_nres_nb_mse.png")
plt.savefig(output_path)

# Plot for Non-Resampled Data (Test Loss)
plt.figure(figsize=(8, 6))
# Scatter for points without resampling (white border, no circle marker)
scatter_non_resampling_testloss = plt.scatter(nres_non_resampled, nb_non_resampled, s=280, 
                                              c=testloss_non_resampled, cmap='inferno_r', edgecolor='w', 
                                              linewidths=1, marker='o', facecolors='none')
cbar = plt.colorbar(scatter_non_resampling_testloss)
cbar.set_label("Test Loss", fontsize=legend_fs, fontweight="bold")
plt.xlabel(r"$n_{\mathrm{res}}$", fontweight="bold", fontsize=axis_fs)
plt.ylabel(r"$n_{\mathrm{b}}$", fontweight="bold", fontsize=axis_fs)
plt.title("Test Loss", fontweight="bold", fontsize=title_fs)
cbar.ax.tick_params(labelsize=tick_fs)
plt.tick_params(axis='both', which='both', labelsize=tick_fs)
plt.grid(True)
plt.tight_layout()

# Save plot for Non-Resampled Test Loss
output_path = os.path.join(output_folder, f"{pinn}_nres_nb_testloss.png")
plt.savefig(output_path)

# Plot for Non-Resampled Data (MSE)
plt.figure(figsize=(8, 6))
# Scatter for points without resampling (white border, no circle marker)
scatter_non_resampling_mse = plt.scatter(nres_non_resampled, nb_non_resampled, s=280, 
                                         c=mse_non_resampled, cmap='viridis_r', edgecolor='w', 
                                         linewidths=1, marker='o', facecolors='none')
cbar = plt.colorbar(scatter_non_resampling_mse)
cbar.set_label("MSE", fontsize=legend_fs, fontweight="bold")
plt.xlabel(r"$n_{\mathrm{res}}$", fontweight="bold", fontsize=axis_fs)
plt.ylabel(r"$n_{\mathrm{b}}$", fontweight="bold", fontsize=axis_fs)
plt.title("MSE", fontweight="bold", fontsize=title_fs)
cbar.ax.tick_params(labelsize=tick_fs)
cbar.ax.yaxis.get_offset_text().set_fontsize(tick_fs)
plt.tick_params(axis='both', which='both', labelsize=tick_fs)
plt.grid(True)
plt.tight_layout()

# Save plot for Non-Resampled MSE
output_path = os.path.join(output_folder, f"{pinn}_nres_nb_mse.png")
plt.savefig(output_path)

# Show all plots
plt.show()