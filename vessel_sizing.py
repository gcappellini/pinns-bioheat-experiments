import numpy as np
import os
import matplotlib.pyplot as plt

current_file = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file)
output_dir = f"{script_directory}/vessel"
os.makedirs(output_dir, exist_ok=True)

# Constants
rho = 1000  # Density of water in kg/m^3
mu = 8.9e-4  # Dynamic viscosity of water in kg/(m·s)
Re_limit = 2300  # Reynolds number limit for laminar flow

# Diameter range (converted to mm)
D_range_mm = np.linspace(0.5, 10, 100)  # Diameter range from 1 mm to 100 mm

# Convert diameter range to meters
D_range = D_range_mm / 1000  # Convert mm to m for calculations

# Calculate velocity for laminar flow (converted to cm/s)
v_laminar_cm_s = (Re_limit * mu * 100) / (rho * D_range)  # Convert m/s to cm/s

# Plot
plt.plot(D_range_mm, v_laminar_cm_s)
plt.xlabel('Diameter (mm)')
plt.ylabel('Velocity (cm/s)')
plt.title('Velocity vs. Diameter for Laminar Flow of Water')
plt.grid(True)
plt.savefig(f'{output_dir}/sizing.png')
plt.show()