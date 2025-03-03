import utils_meas as utils
import os
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import v1v2_calc as cc
from uncertainties import ufloat
from scipy.interpolate import interp1d

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

# Constants
K = 15
a4 = -2
b2 = 1

# List of test indices
nn = [0, 1, 2, 3]

# Create a single figure with 4 subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # 2x2 grid of subplots

for idx, n in enumerate(nn):
    # Import test data
    a = utils.import_testdata(n)

    # Extract variables from data
    y10 = a[0:4][0, 2]
    y20 = a[0:4][3, 2]
    y30 = a[0][-1]
    a5 = cc.a5


    # Define the sinusoidal interpolation function
    def sin_interpol(x, a5, y10, y20, y30, K, b2):
        # Calculate b1 using given formula
        b1 = (a5 * (y30 - y20) + K * (y20 - y10)) / (K * np.sin(b2) + b2 * np.cos(b2))
        return y10 + b1 * np.sin(b2 * x)

    # Define the quadratic interpolation function
    def quad_interpol(x, a5, y10, y20, y30, K, a4):
        # Calculate b1 using given formula
        b1 = (a5 * y30 + (K - a5) * y20 - (2 + K) * a4) / (1 + K)
        return y10 + b1 * x + a4 * x**2

    # Define x values for interpolation
    x = np.linspace(0, 1, num=10)

    # Compute interpolation values
    sin_values = sin_interpol(x, a5, y10, y20, y30, K, b2)
    quad_values = quad_interpol(x, a5, y10, y20, y30, K, a4)

    # Determine subplot location
    ax = axs[idx // 2, idx % 2]  # Compute subplot index

    # Plot the results on the appropriate subplot
    ax.plot(x, sin_values, label=f'Sinusoidal (b2={b2})', color='blue')
    ax.plot(x, quad_values, label=f'Quadratic (a4={a4})', color='red')

    # Add titles and labels
    ax.set_title(f'Test Case {n}')
    ax.set_xlabel('x')
    ax.set_ylabel('ic(x)')
    ax.legend()
    ax.grid(True)

# Adjust layout for better spacing between subplots
plt.tight_layout()



# Save the complete figure
plt.savefig(f"{src_dir}/data/measurements/vessel/ic_tests_combined.png", dpi=120)
# Display the complete figure with all subplots
plt.show()
plt.clf()  # Clear the figure after saving