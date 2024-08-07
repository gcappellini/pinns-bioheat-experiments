import numpy as np
import matplotlib.pyplot as plt
import os
import matlab.engine

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
T_fluid_initial = 37  # initial fluid temperature in °C

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
    
    # df = pd.DataFrame({'x':x, 't_fluid':T_fluid, 't_wall':T_wall, 'Q': Q})
    df = np.vstack((x, T_fluid, Q))

    return df

def scale_t(t):
    return (t - T_fluid_initial) / (T_tumour - T_fluid_initial)


# Define the temperature_distribution function
def temperature_distribution(v, R1, xcc):
    r = np.linspace(-R2/2, R2/2, 100) 
    # Calculate normalized temperature difference
    df1 = vessel_temperature_distribution(v, R1)
    x_vals, t_wall_vals, q = df1[0], df1[1], df1[2]

    x_index = np.argmin(np.abs(x_vals - xcc))
    T_tissue = t_wall_vals[x_index] + q[x_index] * ((np.log(np.abs(r) / R1) / (2 * np.pi * k * dx)))
    T = np.where(np.abs(r)<=R1, t_wall_vals[x_index], T_tissue)
    return T



# Vessel 1 Parameters
R1_vessel1 = 0.5 / 1000  # radius in meters
v_vessel1 = 1.0 / 100  # velocity in m/s
R2 = 7.0 / 100  # radius of the tumour in meters

# Vessel 2 Parameters
R1_vessel2 = 1.0 / 1000  # radius in meters
v_vessel2 = 1.5 / 100  # velocity in m/s

# Vessel 3 Parameters
R1_vessel3 = 1.5 / 1000  # radius in meters
v_vessel3 = 2.0 / 100  # velocity in m/s

# Calculate temperature distribution for each vessel
dd1 = vessel_temperature_distribution(v_vessel1, R1_vessel1)
dd2 = vessel_temperature_distribution(v_vessel2, R1_vessel2)
dd3 = vessel_temperature_distribution(v_vessel3, R1_vessel3)

temp_vessel1, temp_vessel2, temp_vessel3 = dd1[1], dd2[1], dd3[1]

temp_ratio_vessel1 = scale_t(temp_vessel1)
temp_ratio_vessel2 = scale_t(temp_vessel2)
temp_ratio_vessel3 = scale_t(temp_vessel3)


# Grid settings
xgr = 25/100  # x-axis (length along vessel in cm)
r = np.linspace(-R2, R2, 100)  # radial axis (from -1 cm to 1 cm)

# Calculate normalized temperature difference
T_distribution_v1 = temperature_distribution(v_vessel1, R1_vessel1, xgr)
T_normalized_v1 = scale_t(T_distribution_v1)

T_distribution_v2 = temperature_distribution(v_vessel2, R1_vessel2, xgr)
T_normalized_v2 = scale_t(T_distribution_v2)

T_distribution_v3 = temperature_distribution(v_vessel3, R1_vessel3, xgr)
T_normalized_v3 = scale_t(T_distribution_v3)

# print(T_normalized_v1[0], T_normalized_v2[0], T_normalized_v3[0])


eng = matlab.engine.start_matlab()
eng.cd(src_dir, nargout=0)
eng.simple_script(nargout=0)
eng.quit()
a = np.loadtxt(f"{src_dir}/output_pbhe.txt")
r_values_matlab = a[:, 0]
vessel1_matlab = a[:, 1]
vessel2_matlab = a[:, 2]
vessel3_matlab = a[:, 3]


# Plot results
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

# Plot the temperature distribution
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.plot(r*100, T_normalized_v1, color='C0', label='Vessel 1', linewidth=1.5)
ax.plot(r_values_matlab*100, vessel1_matlab, color='C0', label='Perfusion 1', linewidth=1.2, linestyle="--")
ax.plot(r*100, T_normalized_v2, color='C1', label='Vessel 2', linewidth=1.5)
ax.plot(r_values_matlab*100, vessel2_matlab, color='C1', label='Perfusion 2', linewidth=1.2, linestyle="--")
ax.plot(r*100, T_normalized_v3, color='C2', label='Vessel 3', linewidth=1.5)
ax.plot(r_values_matlab*100, vessel3_matlab, color='C2', label='Perfusion 3', linewidth=1.2, linestyle="--")
# Customize the plot
ax.set_xlabel('Radial Distance, $r$ (cm)', fontsize=12)
ax.set_ylabel(r'$\frac{T_{\text{tissue}}(x, r) - T_{\text{fluid}}(0)}{T_{\text{tumour}} - T_{\text{fluid}}(0)}$', fontsize=12)
ax.set_title(f'Temperature Distribution Along Radius, x={xgr} m', fontsize=14)

ax.legend()
plt.grid(True)
plt.savefig(f"{src_dir}/data/simulations/t_ratio_along_radius.png", dpi=120)
plt.show()