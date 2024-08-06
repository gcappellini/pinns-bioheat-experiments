import numpy as np
import matplotlib.pyplot as plt
import os
import ratio_along_vessel as ral

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)


# Constants
L = 0.15  # height of the phantom in meters
N = 100  # number of steps
k = 0.6  # conductivity of tissue

cfl = 4186  # specific heat capacity of fluid in J/kg°C
rho = 1000  # density of fluid in kg/m^3
T_tumour = 42  # tumour temperature in °C
T_fluid_initial = 37  # minimum temperature in °C

# Calculate step size
dx = L / N
x = np.linspace(0, L, N)
R2 = 7/100
r = np.linspace(0, R2, 100)

# Function to calculate temperature distribution for a vessel
def temperature_distribution(v, R1, xcc):
    # Heat transfer coefficient (h)
    h = 3.66 * k / (2 * R1)

    # Resistances
    R_w = 1 / (2 * np.pi * R1 * h * dx)
    R_t = np.log(R2 / R1) / (2 * np.pi * k * dx)
    R_f = 1 / (np.pi * (R1 ** 2) * v * cfl * rho)

    # Arrays to store temperature values
    T_fluid = np.zeros(N)
    Q = np.zeros(N)

    # Set initial values
    T_fluid[0] = T_fluid_initial
    Q[0] = (T_tumour-T_fluid[0])/(R_f+R_w+R_t)

    # Iterative computation
    for i in range(N - 1):
        # Update fluid temperature
        T_fluid[i + 1] = T_fluid[i] + Q[i]*R_f
        Q[i+1] = (T_tumour-T_fluid[i + 1])/(R_f+R_w+R_t)

    x_index = np.argmin(np.abs(x - xcc))
    t_fluid = T_fluid[x_index]
    q = Q[x_index]

    T_wall1 = t_fluid - q*R_w
    T_wall2 = t_fluid + q*R_w

    T_tissue1 = T_wall1 - T_wall1 * ((np.log((0.00001/np.abs(r))) / (2 * np.pi * k * dx)))
    T_tissue2 = T_wall2 + q * ((np.log(np.abs(r) / (R2/2 + R1)) / (2 * np.pi * k * dx)))

    T = np.where(r <= R2/2 - R1, T_tissue1,
                 np.where((R2/2 - R1 < r) & (r <= R2/2 + R1), T_fluid,
                          T_tissue2))
    return T

def scale_t(t):
    return (t - T_fluid_initial) / (T_tumour - T_fluid_initial)

# Vessel 1 Parameters
R1_vessel1 = 0.5 / 1000  # radius in meters
v_vessel1 = 1.0 / 100  # velocity in m/s
# R2 = 4.0 / 100  # radius of the tumour in meters

# Vessel 2 Parameters
R1_vessel2 = 1.0 / 1000  # radius in meters
v_vessel2 = 1.5 / 100  # velocity in m/s

# Vessel 3 Parameters
R1_vessel3 = 1.5 / 1000  # radius in meters
v_vessel3 = 2.0 / 100  # velocity in m/s

# Grid settings
xgr = 25/100  # x-axis (length along vessel in cm)

# Calculate normalized temperature difference
T_distribution_v1 = temperature_distribution(v_vessel1, R1_vessel1, xgr)
T_normalized_v1 = scale_t(T_distribution_v1)

T_distribution_v2 = temperature_distribution(v_vessel2, R1_vessel2, xgr)
T_normalized_v2 = scale_t(T_distribution_v2)

T_distribution_v3 = temperature_distribution(v_vessel3, R1_vessel3, xgr)
T_normalized_v3 = scale_t(T_distribution_v3)

# a = np.loadtxt(f"{src_dir}/output_pbhe.txt")
# r_values_matlab = a[:, 0]
# vessel1_matlab = a[:, 1]
# vessel2_matlab = a[:, 2]
# vessel3_matlab = a[:, 3]

# print(np.square(T_normalized_v2 - vessel2_matlab[:-1]).mean())

# Plot results
plt.figure(figsize=(10, 5))
x_values = np.linspace(0, 100 * L, N)


# Plot the temperature distribution
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.plot(x*100, T_normalized_v1, color='C0', label='Vessel 1', linewidth=1.5)
# ax.plot(r_values_matlab, vessel1_matlab, color='C0', label='Perfusion 1', linewidth=1.2, linestyle="--")
ax.plot(x*100, T_normalized_v2, color='C1', label='Vessel 2', linewidth=1.5)
# ax.plot(r_values_matlab, vessel2_matlab, color='C1', label='Perfusion 2', linewidth=1.2, linestyle="--")
ax.plot(x*100, T_normalized_v3, color='C2', label='Vessel 3', linewidth=1.5)
# ax.plot(r_values_matlab, vessel3_matlab, color='C2', label='Perfusion 3', linewidth=1.2, linestyle="--")

# Customize the plot
ax.set_xlabel('Radial Distance, $r$ (cm)', fontsize=12)
ax.set_ylabel(r'$\frac{T_{\text{tissue}}(x, r) - T_{\text{fluid}}(0)}{T_{\text{tumour}} - T_{\text{fluid}}(0)}$', fontsize=12)
ax.set_title(f'Temperature Distribution Along Radius, x={xgr} m', fontsize=14)

ax.legend()
plt.grid(True)
plt.savefig(f"{src_dir}/data/simulations/t_ratio_along_radius.png", dpi=120)
plt.show()
plt.clf()