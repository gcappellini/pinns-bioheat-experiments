import numpy as np
import matplotlib.pyplot as plt
import os
import matlab.engine
import json
import import_vessel_data as ivd
import sympy as sp
from sympy.solvers import solve


current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

with open(f"{src_dir}/properties.json", 'r') as f:
  data = json.load(f)

# Constants
L, L0 = data["L"], data["L0"]
k, cfl, rho, h = data["k"], data["c"], data["rho"], data["h"]

T_tumour, Ttis, Tw = data["Tmax"], data["Ttis"], data["Tw"] # tumour temperature in °C
T_fluid_initial = Ttis

P0, d, a = data["P0"], data["d"], data["a"]

x_tc = np.linspace(0, 1, 8).round(2)* 0.15


def scale_t(t):
    return (t - T_fluid_initial) / (T_tumour - T_fluid_initial)

def rescale_t(t):
    return T_fluid_initial + t*(T_tumour - T_fluid_initial)


def integrated_pde(x, C1, C2):
    flux = P0 * np.exp(-a*x)+C1
    temperature = (1/k)*(-a*P0*np.exp(-a*x)+C1*x+C2)
    return flux, temperature


def steady_rv(r, v):
    # r1_vessel2 = 1.0 / 1000
    xw1 = x_tc[2]-r
    xw2 = x_tc[2]+r

    hv = 3.66 * k / (2 * r)

    Tfl = calculate_tfl(L/2, r, v)

    # Define the temperature distribution function T(x)
    def temperature_distribution_1(x, C1_var, C2_var):

        flux_0, t_0 = integrated_pde(0, C1_var, C2_var)
        flux_xw1, t_xw1 = integrated_pde(xw1, C1_var, C2_var)

        eq1= sp.Eq(flux_0, h*(Tw-t_0)) 
        eq2 = sp.Eq(flux_xw1, -hv*(t_xw1-Tfl))

        # Solve for C1 and C2
        solution = sp.solve([eq1, eq2], (C1_var, C2_var))
        C1_solved, C2_solved = solution[C1_var], solution[C2_var]

        array = []
        for el in x:
            _, theta = integrated_pde(el, C1_solved, C2_solved)
            array.append(theta)

        return np.array(array)


    def temperature_distribution_2(x, C1_var, C2_var):

        flux_xw2, t_xw2 = integrated_pde(xw2, C1_var, C2_var)
        _, t_L0 = integrated_pde(L0, C1_var, C2_var)

        eq1= sp.Eq(flux_xw2, hv*(t_xw2-Tfl)) 
        eq2 = sp.Eq(t_L0, Ttis)

        solution = sp.solve([eq1, eq2], (C1_var, C2_var))
        C1_solved, C2_solved = solution[C1_var], solution[C2_var]

        array = []
        for el in x:
            _, theta = integrated_pde(el, C1_solved, C2_solved)
            array.append(theta)

        return np.array(array)

    def temperature_distribution(x_arr, C1_in, C2_in):

        T = np.where(x_arr <= xw1, temperature_distribution_1(x_arr, C1_in, C2_in),
                    np.where((xw1 < x_arr) & (x_arr <= xw2), Tfl,
                            np.where((xw2 < x_arr) & (x_arr <= L0), temperature_distribution_2(x_arr, C1_in, C2_in),
                                    Ttis
                            )))
        return T

# Calculate step size
N = 100  # number of steps
dy = L / N
y = np.linspace(0, L, N)


# Generate depth values (x) and calculate temperature
depth = np.linspace(0, 3*L0, 100) 

C1_ver, C2_ver = sp.symbols('C1 C2', real=True)
temperature = temperature_distribution(depth, C1_ver, C2_ver)

# eng = matlab.engine.start_matlab()
# eng.cd(src_dir, nargout=0)
# eng.simple_script(nargout=0)
# eng.quit()
# a = np.loadtxt(f"{src_dir}/output_pbhe.txt")
# x_values_matlab = a[:, 0]*L0
# vessel1_matlab = a[:, 1]
# vessel2_matlab = a[:, 2]
# vessel3_matlab = a[:, 3]

file_path = f"{src_dir}/data/vessel/20240522_1.txt"  # Replace with your file path
timeseries_data = ivd.load_measurements(file_path)
meas1 = ivd.extract_entries(timeseries_data, 2*60, 3*60)
meas2 = ivd.extract_entries(timeseries_data, 30*60, 35*60)
meas3 = ivd.extract_entries(timeseries_data, 56*60, 58*60)
meas4 = ivd.extract_entries(timeseries_data, 83*60, 85*60)

x_meas = np.array([x_tc[0],x_tc[1],x_tc[4],x_tc[7]])
y_meas1 = np.array([meas1["y2"],meas1["gt2"],meas1["gt1"],meas1["y1"]])
y_meas2 = np.array([meas2["y2"],meas2["gt2"],meas2["gt1"],meas2["y1"]])
y_meas3 = np.array([meas3["y2"],meas3["gt2"],meas3["gt1"],meas3["y1"]])
y_meas4 = np.array([meas4["y2"],meas4["gt2"],meas4["gt1"],meas4["y1"]])



# Plotting
plt.figure(figsize=(8, 6))
plt.plot(depth * 100, temperature, label="MW Heating Distribution")
# plt.plot(x_values_matlab * 100, rescale_t(vessel1_matlab), label="PBHE W1")
# plt.plot(x_values_matlab * 100, rescale_t(vessel2_matlab), label="PBHE W2")
# plt.plot(x_values_matlab * 100, rescale_t(vessel3_matlab), label="PBHE W3")

plt.plot(x_meas * 100, y_meas1, label="meas 1")
plt.plot(x_meas * 100, y_meas2, label="meas 2")
plt.plot(x_meas * 100, y_meas3, label="meas 3")
plt.plot(x_meas * 100, y_meas4, label="meas 4")

# plt.plot(x_values_matlab * 100, rescale_t(vessel3_matlab), label="PBHE W3")
# plt.xlabel("Depth (cm)")
# plt.ylabel("Temperature (°C)")

ytext = min(temperature)
plt.axvline(100*x_tc[0], color='r', linestyle='--', linewidth=0.8)
plt.text(100 * x_tc[0], ytext, 'y2', fontsize=12, color='r')

plt.axvline(100*x_tc[1], color='r', linestyle='--', linewidth=0.8)
plt.text(100 * x_tc[1], ytext, 'gt2', fontsize=12, color='r')

plt.axvline(100*x_tc[2], color='r', linestyle='--', linewidth=0.8)
plt.text(100 * x_tc[2], ytext, 'w', fontsize=12, color='r')

plt.axvline(100*x_tc[4], color='r', linestyle='--', linewidth=0.8)
plt.text(100 * x_tc[4], ytext, 'gt1', fontsize=12, color='r')

plt.axvline(100*x_tc[7], color='r', linestyle='--', linewidth=0.8)
plt.text(100 * x_tc[7], ytext, 'y1', fontsize=12, color='r')

plt.title("Temperature Distribution in Phantom")
plt.grid(True)
plt.legend()
plt.show()




# # Function to calculate temperature distribution for a vessel
# def vessel_temperature_distribution(v, R1):
#     # Heat transfer coefficient (h)
#     h_flow = 3.66 * k / (2 * R1)
#     R2 = 2.25/100

#     # Resistances
#     R_w = 1 / (2 * np.pi * R1 * h_flow * dx)
#     R_t = np.log(R2 / R1) / (2 * np.pi * k * dx)
#     R_f = 1 / (np.pi * (R1 ** 2) * v * cfl * rho)


#     # Arrays to store temperature values
#     T_fluid = np.zeros(N)
#     T_wall = np.zeros(N)
#     Q = np.zeros(N)

#     # Set initial values
#     T_fluid[0] = T_fluid_initial
#     Q[0] = (T_tumour-T_fluid[0])/(R_f+R_w+R_t)
#     T_wall[0] = T_fluid[0] + Q[0]*R_w

#     # Iterative computation
#     for i in range(N - 1):
#         # Update fluid temperature
#         T_fluid[i + 1] = T_fluid[i] + Q[i]*R_f
#         Q[i+1] = (T_tumour-T_fluid[i + 1])/(R_f+R_w+R_t)
#         T_wall[i + 1] = T_fluid[i+1] + Q[i+1]*R_f
    
#     df = np.vstack((x, T_fluid, T_wall, Q))

#     return df



# def temperature_distribution(v, R1, xcc):
#     r = np.linspace(0, L0, 100) 
#     # Calculate normalized temperature difference
#     df1 = vessel_temperature_distribution(v, R1)
#     x_vals, t_fluid, t_wall, q = df1[0], df1[1], df1[2], df1[3]

#     x_index = np.argmin(np.abs(x_vals - xcc))
#     T_fluid = t_fluid[x_index]
#     T_wall = t_wall[x_index]
#     r_trnsl = r - yv
#     T_tissue = T_wall + q[x_index] * ((np.log(np.abs(r_trnsl) / R1) / (2 * np.pi * k * dx)))

#     T = np.where(r <= yv - R1, T_fluid_initial +(r/yv)*(T_fluid - T_fluid_initial),
#                 np.where((yv - R1 < r) & (r <= yv + R1), T_fluid,
#                         T_tissue))
#     return T



# # Vessel 1 Parameters
# R1_vessel1 = 0.5 / 1000  # radius in meters
# v_vessel1 = 1.0 / 100  # velocity in m/s

# # Vessel 2 Parameters
# R1_vessel2 = 1.0 / 1000  # radius in meters
# v_vessel2 = 1.5 / 100  # velocity in m/s

# # Vessel 3 Parameters
# R1_vessel3 = 1.5 / 1000  # radius in meters
# v_vessel3 = 2.0 / 100  # velocity in m/s
# y_tc = np.linspace(0, 1, 8).round(2)* L0
# yv = y_tc[5]

# # Calculate temperature distribution for each vessel
# dd1 = vessel_temperature_distribution(v_vessel1, R1_vessel1, yv)
# dd2 = vessel_temperature_distribution(v_vessel2, R1_vessel2, yv)
# dd3 = vessel_temperature_distribution(v_vessel3, R1_vessel3, yv)

# temp_vessel1, temp_vessel2, temp_vessel3 = dd1[1], dd2[1], dd3[1]

# temp_ratio_vessel1 = scale_t(temp_vessel1)
# temp_ratio_vessel2 = scale_t(temp_vessel2)
# temp_ratio_vessel3 = scale_t(temp_vessel3)


# # Grid settings
# xgr = 25/100  # x-axis (length along vessel in cm)
# r = np.linspace(0, L0, 100)  # radial axis (from -1 cm to 1 cm)

# # Calculate normalized temperature difference
# T_distribution_v1 = temperature_distribution(v_vessel1, R1_vessel1, xgr, yv)
# T_normalized_v1 = scale_t(T_distribution_v1)

# T_distribution_v2 = temperature_distribution(v_vessel2, R1_vessel2, xgr, yv)
# T_normalized_v2 = scale_t(T_distribution_v2)

# T_distribution_v3 = temperature_distribution(v_vessel3, R1_vessel3, xgr, yv)
# T_normalized_v3 = scale_t(T_distribution_v3)

# # print(T_normalized_v1[0], T_normalized_v2[0], T_normalized_v3[0])



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
# plt.savefig(f"{src_dir}/data/simulations/cart_ratio_along_vessel.png", dpi=120)
# plt.show()

# # Plot the temperature distribution
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111)
# ax.plot(r*100, T_normalized_v1, color='C0', label='Vessel 1', linewidth=1.5)
# ax.plot(x_values_matlab*100, vessel1_matlab, color='C0', label='Perfusion 1', linewidth=1.2, linestyle="--")
# ax.plot(r*100, T_normalized_v2, color='C1', label='Vessel 2', linewidth=1.5)
# ax.plot(x_values_matlab*100, vessel2_matlab, color='C1', label='Perfusion 2', linewidth=1.2, linestyle="--")
# ax.plot(r*100, T_normalized_v3, color='C2', label='Vessel 3', linewidth=1.5)
# ax.plot(x_values_matlab*100, vessel3_matlab, color='C2', label='Perfusion 3', linewidth=1.2, linestyle="--")

# # Add 8 equally spaced dashed vertical lines from 0 to 1
# for i in y_tc:
#     ax.axvline(100*i, color='r', linestyle='--', linewidth=0.8)

# ax.set_xticks(100 * np.array(y_tc))
# ax.set_xticklabels([f"{100 * i:.2f}" for i in y_tc], fontsize=10)
# # Customize the plot
# ax.set_xlabel('Height, $y$ (cm)', fontsize=12)
# ax.set_ylabel(r'$\frac{T_{\text{tissue}}(x, r) - T_{\text{fluid}}(0)}{T_{\text{tumour}} - T_{\text{fluid}}(0)}$', fontsize=12)
# ax.set_title(f'Temperature Distribution Along Height, x={xgr} m', fontsize=14)

# ax.legend()
# plt.grid(True)
# plt.savefig(f"{src_dir}/data/simulations/cart_ratio_along_radius.png", dpi=120)
# plt.show()