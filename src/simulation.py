import numpy as np
import matplotlib.pyplot as plt
import os
import matlab.engine
import json
import import_vessel_data as ivd
import sympy as sp
from sympy.solvers import solve
from scipy import special


current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

with open(f"{src_dir}/properties.json", 'r') as f:
  data = json.load(f)

# Constants
L, L0 = data["L"], data["L0"]
k, cfl, rho, h = data["k"], data["c"], data["rho"], data["h"]

Tmax, Ttis, Tw= data["Tmax"], data["Ttis"], data["Tw"] # tumour temperature in °C

P0, d, a, b, x0, beta = data["P0"], data["d"], data["a"], data["b"], data["x0"], data["beta"]
P0_x = beta * P0 * np.exp(a*x0)

x_tc = np.linspace(0, 1, 8).round(2)* 0.15

Tfl = None

def calculate_tfl(y, v, r):
    global Tfl
    Tfl = np.where(r<=0.0005, Ttis,
                 np.where(np.logical_and(r > 0.0005, r <= 0.001), Ttis+3.5,Ttis+6.4))
    return Tfl

def scale_t(t):
    return (t - Ttis) / (Tmax - Ttis)

def rescale_t(t):
    return Ttis + t*(Tmax - Ttis)


def integrated_pde_x(x, C1, C2):
    flux = P0_x * np.exp(-a*x)+C1
    temperature = (1/k)*(-a*P0_x*np.exp(-a*x)+C1*x+C2)
    return flux, temperature

def temperature_distribution(x):
    C1_var, C2_var  = sp.symbols('C1 C2', real=True)

    flux_0, t_0 = integrated_pde_x(0, C1_var, C2_var)
    _, t_L0 = integrated_pde_x(L0, C1_var, C2_var)

    eq1= sp.Eq(flux_0, h*(Tw-t_0)) 
    eq2 = sp.Eq(t_L0, Ttis)

    # Solve for C1 and C2
    solution = sp.solve([eq1, eq2], (C1_var, C2_var))
    C1_solved, C2_solved = solution[C1_var], solution[C2_var]

    if isinstance(x, int):
        _, theta = integrated_pde_x(x, C1_solved, C2_solved)
    elif isinstance(x, float):
        _, theta = integrated_pde_x(x, C1_solved, C2_solved)
    else:
        thetas = []
        for el in x:
            _, thet = integrated_pde_x(el, C1_solved, C2_solved)
            thetas.append(thet)
        theta = np.array(thetas)

    return theta

xr2 = x_tc[2]-0.01

def t_distribution(x_arr, r, v):
    Tfl = calculate_tfl(0, v, r)
    
    xw2 = x_tc[2]+r
    xw1 = x_tc[2]-r
    T1 = temperature_distribution(x_arr)
    Tr2 = temperature_distribution(xr2)
    hv = 3.66 * k / (2 * r)
    # Qw1 = hv*(temperature_distribution(xw1)-Tfl)
    # ni = 0.1
    # Tw2 = Tfl-ni*Qw1/hv

    T = np.where(x_arr <= xr2, T1,
                    np.where((xr2 < x_arr) & (x_arr <= x_tc[2]), Tr2+(Tfl-Tr2)*np.log(x_arr/xr2)/np.log(xw1/xr2),Tfl + (Ttis-Tfl)*(x_arr-xw2)/(3*L0-xw2)))
    return T

# Calculate step size
N = 100  # number of steps

# Generate depth values (x) and calculate temperature
depth = np.linspace(0, 0.15, 100) 


eng = matlab.engine.start_matlab()
eng.cd(src_dir, nargout=0)
eng.simple_script(nargout=0)
eng.quit()
a = np.loadtxt(f"{src_dir}/output_pbhe.txt")
x_values_matlab = a[:, 0]*L0
pbhe1_matlab = a[:, 1]
pbhe2_matlab = a[:, 2]
pbhe3_matlab = a[:, 3]

file_path = f"{src_dir}/data/vessel/20240522_1.txt"  # Replace with your file path
timeseries_data = ivd.load_measurements(file_path)
meas1 = ivd.extract_entries(timeseries_data, 2*60, 3*60)
meas2 = ivd.extract_entries(timeseries_data, 30*60, 35*60)
meas3 = ivd.extract_entries(timeseries_data, 56*60, 58*60)
meas4 = ivd.extract_entries(timeseries_data, 83*60+4, 83*60+6)

x_meas = np.array([x_tc[0],x_tc[1],x_tc[4],x_tc[7]])
y_meas1 = np.array([meas1["y2"],meas1["gt2"],meas1["gt1"],meas1["y1"]])
y_meas2 = np.array([meas2["y2"],meas2["gt2"],meas2["gt1"],meas2["y1"]])
y_meas3 = np.array([meas3["y2"],meas3["gt2"],meas3["gt1"],meas3["y1"]])
y_meas4 = np.array([meas4["y2"],meas4["gt2"],meas4["gt1"],meas4["y1"]])


# Vessel 1 Parameters
R1_vessel1 = 0.5 / 1000  # radius in meters
v_vessel1 = 1.0 / 100  # velocity in m/s

# Vessel 2 Parameters
R1_vessel2 = 1.0 / 1000  # radius in meters
v_vessel2 = 1.5 / 100  # velocity in m/s

# Vessel 3 Parameters
R1_vessel3 = 1.5 / 1000  # radius in meters
v_vessel3 = 2.0 / 100  # velocity in m/s

# vessel1 = t_distribution(depth, R1_vessel1, v_vessel1)
# vessel2 = t_distribution(depth, R1_vessel2, v_vessel2)
# vessel3 = t_distribution(depth, R1_vessel3, v_vessel3)
# pure = temperature_distribution(depth[:30])

# # # Save the vessel1 array to the npz file
# np.savez(f"{src_dir}/data/simulations/simulation/distr_{R1_vessel1}_{v_vessel1}.npz", vessel1=vessel1)
# np.savez(f"{src_dir}/data/simulations/simulation/distr_{R1_vessel2}_{v_vessel2}.npz", vessel2=vessel2)
# np.savez(f"{src_dir}/data/simulations/simulation/distr_{R1_vessel3}_{v_vessel3}.npz", vessel3=vessel3)
# np.savez(f"{src_dir}/data/simulations/simulation/distr_pure.npz", pure=pure)

# Load the vessel1 array from the npz file
loaded_data1 = np.load(f"{src_dir}/data/simulations/simulation/distr_{R1_vessel1}_{v_vessel1}.npz", allow_pickle=True)
loaded_data2 = np.load(f"{src_dir}/data/simulations/simulation/distr_{R1_vessel2}_{v_vessel2}.npz", allow_pickle=True)
loaded_data3 = np.load(f"{src_dir}/data/simulations/simulation/distr_{R1_vessel3}_{v_vessel3}.npz", allow_pickle=True)
loaded_data4 = np.load(f"{src_dir}/data/simulations/simulation/distr_pure.npz", allow_pickle=True)
vessel1 = loaded_data1['vessel1']
vessel2 = loaded_data2['vessel2']
vessel3 = loaded_data3['vessel3']
pure = loaded_data4['pure']
# Plotting
plt.figure(figsize=(8, 6))
plt.plot(depth * 100, vessel1, label=f"Tfl={calculate_tfl(0, 0, R1_vessel1)}")
plt.plot(depth * 100, vessel2, label=f"Tfl={calculate_tfl(0, 0, R1_vessel2)}")
plt.plot(depth * 100, vessel3, label=f"Tfl={calculate_tfl(0, 0, R1_vessel3)}")
plt.plot(depth[:30] * 100, pure, linestyle="-.", label=f"No Vessel")
plt.plot(x_values_matlab * 100, rescale_t(pbhe1_matlab), label="PBHE W1")
plt.plot(x_values_matlab * 100, rescale_t(pbhe2_matlab), label="PBHE W2")
plt.plot(x_values_matlab * 100, rescale_t(pbhe3_matlab), label="PBHE W3")

plt.plot(x_meas * 100, y_meas1, label="meas 1")
plt.plot(x_meas * 100, y_meas2, label="meas 2")
plt.plot(x_meas * 100, y_meas3, label="meas 3")
plt.plot(x_meas * 100, y_meas4, label="meas 4")

# plt.plot(x_values_matlab * 100, rescale_t(vessel3_matlab), label="PBHE W3")
plt.xlabel("Depth (cm)")
plt.ylabel("Temperature (°C)")

ytext = min(vessel1)
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


# # Plot results
# plt.figure(figsize=(10, 5))

# # Plot each vessel
# plt.plot(100*y, temp_vessel1[1], label='Fluid')
# plt.plot(100*y, temp_vessel1[2], label='Wall 1')
# plt.plot(100*y, temp_vessel1[3], label='Wall 2')

# # Directly label the lines
# plt.text(100 * L - 5, temp_vessel1[1][-1] + 0.01, 'Vessel 1', fontsize=12, color='C0')
# plt.text(100 * L - 5, temp_vessel1[2][-1] + 0.01, 'Vessel 2', fontsize=12, color='C1')
# plt.text(100 * L - 5, temp_vessel1[3][-1] + 0.01, 'Vessel 3', fontsize=12, color='C2')

# plt.xlabel(r'Location along the vessel, $y$ (cm)', fontsize=12)
# # plt.ylabel(r'$\frac{T_{\text{fluid}}(x)-T_{\text{fluid}}(0)}{T_{\text{tumour}}-T_{\text{fluid}}(0)}$', fontsize=15)
# plt.ylabel(r'$T_{\text{fluid}}(y)$', fontsize=15)
# plt.title('Temperature as a Function of Location Along Vessels', fontsize=14)
# plt.grid(True)
# plt.savefig(f"{src_dir}/data/simulations/prova_along_vessel.png", dpi=120)
# plt.show()
