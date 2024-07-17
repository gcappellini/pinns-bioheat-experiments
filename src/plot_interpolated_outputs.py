import utils_meas as utils
import os
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)



n = "measurements/vessel/1"  # Example argument
Xobs = utils.import_obsdata(n)

# Extract original data and interpolated data for plotting
g = utils.import_testdata(n)
instants = np.unique(g[:, 1])

# Real points
real_y1 = g[g[:, 0] == 0.0][:, -2]
real_y2 = g[g[:, 0] == 1.0][:, -2]
real_y3 = g[g[:, 0] == 1.0][:, -1]

# Interpolated points
interp_y1 = utils.f1(instants)
interp_y2 = utils.f2(instants)
interp_y3 = utils.f3(instants)

# Plotting
plt.figure(figsize=(12, 8))

# Plot for f1
plt.subplot(3, 1, 1)
plt.plot(instants, real_y1, 'o', label='Real y1')
plt.plot(instants, interp_y1, '-', label='Interpolated y1')
plt.title('Interpolation vs Real Points for f1')
plt.xlabel('Instants')
plt.ylabel('Values')
plt.legend()

# Plot for f2
plt.subplot(3, 1, 2)
plt.plot(instants, real_y2, 'o', label='Real y2')
plt.plot(instants, interp_y2, '-', label='Interpolated y2')
plt.title('Interpolation vs Real Points for f2')
plt.xlabel('Instants')
plt.ylabel('Values')
plt.legend()

# Plot for f3
plt.subplot(3, 1, 3)
plt.plot(instants, real_y3, 'o', label='Real y3')
plt.plot(instants, interp_y3, '-', label='Interpolated y3')
plt.title('Interpolation vs Real Points for f3')
plt.xlabel('Instants')
plt.ylabel('Values')
plt.legend()

plt.tight_layout()
plt.show()