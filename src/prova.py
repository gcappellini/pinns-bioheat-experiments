import utils_meas as utils
import os
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import coeff_calc as cc
from uncertainties import ufloat
from scipy.interpolate import interp1d

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

import numpy as np
import matplotlib.pyplot as plt

# Constants
y10 = 0.2
y20 = 0.4
y30 = 0.8
a5 = 150
k = 4

# Coefficients based on solution
A = 30
B = -0.2
C = -29.6
D = y10

# Function Definition
def ic(x):
    return A * x**3 + B * x**2 + C * x + D

# Derivative
def dic_dx(x):
    return 3 * A * x**2 + 2 * B * x + C

# Domain
x_values = np.linspace(0, 1, 100)
y_values = ic(x_values)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label='ic(x)', color='blue')
plt.axhline(0, color='grey', linestyle='--')
plt.axhline(1, color='grey', linestyle='--')
plt.scatter(0, y10, color='red', label='ic(0) = y10')
plt.scatter(1, ic(1), color='green', label='ic(1) = calculated')
plt.title('Plot of ic(x) with Boundary Conditions')
plt.xlabel('x')
plt.ylabel('ic(x)')
plt.legend()
plt.grid(True)
plt.show()