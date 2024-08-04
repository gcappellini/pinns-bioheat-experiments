import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)


# Constants
L = 0.5  # length of the vessel in meters
N = 100  # number of steps
k = 0.6  # conductivity of tissue

cfl = 4186  # specific heat capacity of fluid in J/kg°C
rho = 1000  # density of fluid in kg/m^3
T_tumour = 42  # tumour temperature in °C

# Initial conditions
T_blood_initial = 37  # initial blood temperature in °C

# Calculate step size
dx = L / N
x = np.linspace(0, L, N)

# Function to calculate temperature distribution for a vessel
def vessel_temperature_distribution(v, R1):
    # Heat transfer coefficient (h)
    h = 3.66 * k / (2 * R1)

    # Resistances
    R_w = 1 / (2 * np.pi * R1 * h * dx)
    R_t = np.log(R2 / R1) / (2 * np.pi * k * dx)
    R_f = 1 / (np.pi * (R1 ** 2) * v * cfl * rho)

    # Arrays to store temperature values
    T_blood = np.zeros(N)
    T_wall = np.zeros(N)
    Q = np.zeros(N)

    # Set initial values
    T_blood[0] = T_blood_initial
    Q[0] = (T_tumour-T_blood[0])/(R_f+R_w+R_t)
    T_wall[0] = R_w*Q[0] + T_blood[0]

    # Iterative computation
    for i in range(N - 1):
        # Update fluid temperature
        T_blood[i + 1] = T_blood[i] + Q[i]*R_f
        Q[i+1] = (T_tumour-T_blood[i + 1])/(R_f+R_w+R_t)
        T_wall[i + 1] = R_w*Q[i+1] + T_blood[i+1]
    
    df = pd.DataFrame({'x':x, 't_blood':T_blood, 't_wall':T_wall, 'Q': Q})

    return df

def scale_t(t):
    return (t - T_blood_initial) / (T_tumour - T_blood_initial)

def temp_tissue(v, R1):
    data = vessel_temperature_distribution(v, R1)
    x, t_wall, qq = data['x'], data['t_wall'], data['Q']
    r = np.linspace(R1, R2, num=N)
    T_tissue = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            T_tissue[i, j] = t_wall[i] + qq[i] * ((np.log(r[j] / R1) / (2 * np.pi * k * dx)))

    # Flatten the arrays for DataFrame creation
    x_flat = np.repeat(x, N)  # Repeat x values for each radial position
    r_flat = np.tile(r, N)  # Tile r values for each axial position
    T_tissue_flat = T_tissue.flatten()  # Flatten temperature array

    # Create the DataFrame
    df = pd.DataFrame({
        'x': x_flat, 
        'r': r_flat,  
        't_tissue': T_tissue_flat
    })
    return df

# Define the temperature_distribution function
def temperature_distribution(v, R1, x, r):
    # Get DataFrames
    df1 = vessel_temperature_distribution(v, R1)
    df2 = temp_tissue(v, R1)
    
    # Find the closest x value index
    x_index = (np.abs(df1['x'] - x)).idxmin()

    # Check radial position
    if 0 <= r < R1:
        # Return blood temperature for r in [0, R1)
        return df1['t_blood'].iloc[x_index]
    elif r == R1:
        # Return wall temperature for r = R1
        return df1['t_wall'].iloc[x_index]
    elif R1 < r <= R2:
        # Find the closest r value index for tissue
        df2_filtered = df2[df2['x'] == df1['x'].iloc[x_index]]
        r_index = (np.abs(df2_filtered['r'] - r)).idxmin()
        return df2['t_tissue'].iloc[r_index]



# Vessel 1 Parameters
R1_vessel1 = 0.5 / 1000  # radius in meters
v_vessel1 = 1.0 / 100  # velocity in m/s
R2 = 2.0 / 100  # radius of the tumour in meters

# Vessel 2 Parameters
R1_vessel2 = 1.0 / 1000  # radius in meters
v_vessel2 = 1.5 / 100  # velocity in m/s

# Vessel 3 Parameters
R1_vessel3 = 1.5 / 1000  # radius in meters
v_vessel3 = 2.0 / 100  # velocity in m/s

# Calculate temperature distribution for each vessel
temp_vessel1, q_vessel1 = vessel_temperature_distribution(v_vessel1, R1_vessel1)
temp_vessel2, q_vessel2 = vessel_temperature_distribution(v_vessel2, R1_vessel2)
temp_vessel3, q_vessel3 = vessel_temperature_distribution(v_vessel3, R1_vessel3)

temp_ratio_vessel1 = scale_t(temp_vessel1)
temp_ratio_vessel2 = scale_t(temp_vessel2)
temp_ratio_vessel3 = scale_t(temp_vessel3)

# # Plot results
# plt.figure(figsize=(10, 5))
# x_values = np.linspace(0, 100 * L, N)

# # Plot each vessel
# plt.plot(x_values, temp_ratio_vessel1, label='Vessel 1')
# plt.plot(x_values, temp_ratio_vessel2, label='Vessel 2')
# plt.plot(x_values, temp_ratio_vessel3, label='Vessel 3')

# # Directly label the lines
# plt.text(100 * L - 5, temp_ratio_vessel1[-1] + 0.01, 'Vessel 1', fontsize=12, color='C0')
# plt.text(100 * L - 5, temp_ratio_vessel2[-1] + 0.01, 'Vessel 2', fontsize=12, color='C1')
# plt.text(100 * L - 5, temp_ratio_vessel3[-1] + 0.01, 'Vessel 3', fontsize=12, color='C2')

# plt.xlabel(r'Location along the vessel, $x$ (cm)', fontsize=12)
# plt.ylabel(r'$\frac{T_{\text{fluid}}(x)-T_{\text{fluid}}(0)}{T_{\text{tumour}}-T_{\text{fluid}}(0)}$', fontsize=15)
# plt.title('Temperature Ratio as a Function of Location Along Vessels', fontsize=14)
# plt.grid(True)
# plt.savefig(f"{src_dir}/data/simulations/t_ratio_along_vessel.png", dpi=120)
# plt.show()

grid = temp_tissue(v_vessel2, R1_vessel2)

print(grid.shape)

# Grid settings
Nx = 100  # number of points along the x-axis
Nr = 100  # number of points along the radial direction
x = np.linspace(0, 50, Nx)  # x-axis (length along vessel in cm)
r = np.linspace(-1.0, 1.0, Nr)  # radial axis (from -1 cm to 1 cm)
X, R = np.meshgrid(x, r)  # create meshgrid for plotting

# Calculate normalized temperature difference
T_distribution = temperature_distribution(X, R)
T_normalized = (T_distribution - T_blood_initial) / norm_denominator

# Plot the 3D temperature distribution
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(R*100, X, T_normalized, cmap='viridis', edgecolor='none', alpha=0.8)

# Customize the plot
ax.set_xlabel('Radial Distance, $r$ (cm)', fontsize=12)
ax.set_ylabel('Location Along Vessel, $x$ (cm)', fontsize=12)
ax.set_zlabel(r'$\frac{T_{\text{tissue}}(x, r) - T_{\text{fluid}}(0)}{T_{\text{tumour}} - T_{\text{fluid}}(0)}$', fontsize=12)
ax.set_title('3D Temperature Distribution in Vessel 2 Plane', fontsize=14)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Normalized Temperature Difference')
plt.show()