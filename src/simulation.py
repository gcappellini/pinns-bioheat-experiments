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

T_tumour, Ttis, Tw = data["Tmax"], data["Ttis"], data["Tw"] # tumour temperature in °C
T_fluid_initial = Ttis
T_fluid_end = Ttis + .80

P0, d, a, b, x0, beta = data["P0"], data["d"], data["a"], data["b"], data["x0"], data["beta"]
P0_x = beta* P0 * np.exp(a*x0)

x_tc = np.linspace(0, 1, 8).round(2)* 0.15


def scale_t(t):
    return (t - T_fluid_initial) / (T_tumour - T_fluid_initial)

def rescale_t(t):
    return T_fluid_initial + t*(T_tumour - T_fluid_initial)


def integrated_pde_x(x, C1, C2):
    flux = P0_x * np.exp(-a*x)+C1
    temperature = (1/k)*(-a*P0_x*np.exp(-a*x)+C1*x+C2)
    return flux, temperature


def calculate_tw1(yps, ra):

    hv = 3.66 * k / (2 * ra)
    xw1 = x_tc[2]-ra
    P0_y = beta*np.exp(-a*(xw1-x0))*P0

    def integrated_pde_y(y, C1, C2):
        flux = - P0_y * (np.sqrt(np.pi)/2)*special.erf(np.sqrt(b)*y)+C1
        temperature = (1/k)* (-P0_y)* (np.sqrt(np.pi)/2)*(np.sqrt(b)*y*special.erf(np.sqrt(b)*y)+np.exp(-b*y**2)/np.sqrt(np.pi)+C2)+C1*y
        return flux, temperature

    C1_v, C2_v = sp.symbols('C1 C2', real=True)
    fl_start, t_start = integrated_pde_y(-L/2, C1_v, C2_v)
    fl_end, t_end = integrated_pde_y(L/2, C1_v, C2_v)

    eq1= sp.Eq(fl_start, hv*(t_start-T_fluid_initial)) 
    eq2 = sp.Eq(fl_end, hv*(t_end-T_fluid_end))

    # Solve for C1 and C2
    solution = sp.solve([eq1, eq2], (C1_v, C2_v))
    C1_solved, C2_solved = solution[C1_v], solution[C2_v]

    if isinstance(yps, int):
        _, theta = integrated_pde_y(yps, C1_solved, C2_solved)
    if isinstance(yps, float):
        _, theta = integrated_pde_y(yps, C1_solved, C2_solved)
    else:
        thetas = []
        for el in yps:
            _, thet = integrated_pde_y(el, C1_solved, C2_solved)
            thetas.append(thet)
        theta = np.array(thetas)

    return theta

def calculate_tw2(yps, ra):

    hv = 3.66 * k / (2 * ra)
    xw2 = x_tc[2]+ra
    P0_y2 = beta*np.exp(-a*(xw2-x0))*P0

    def integrated_pde_y(y, C1, C2):
        flux = - P0_y2 * (np.sqrt(np.pi)/2)*special.erf(np.sqrt(b)*y)+C1
        temperature = (1/k)* (-P0_y2)* (np.sqrt(np.pi)/2)*(np.sqrt(b)*y*special.erf(np.sqrt(b)*y)+np.exp(-b*y**2)/np.sqrt(np.pi)+C2)+C1*y
        return flux, temperature

    C1_v, C2_v = sp.symbols('C1 C2', real=True)
    fl_start, t_start = integrated_pde_y(-L/2, C1_v, C2_v)
    fl_end, t_end = integrated_pde_y(L/2, C1_v, C2_v)

    eq1= sp.Eq(fl_start, hv*(T_fluid_initial-t_start)) 
    eq2 = sp.Eq(fl_end, hv*(T_fluid_end-t_end))

    # Solve for C1 and C2
    solution = sp.solve([eq1, eq2], (C1_v, C2_v))
    C1_solved, C2_solved = solution[C1_v], solution[C2_v]

    if isinstance(yps, int):
        _, theta = integrated_pde_y(yps, C1_solved, C2_solved)
    if isinstance(yps, float):
        _, theta = integrated_pde_y(yps, C1_solved, C2_solved)
    else:
        thetas = []
        for el in yps:
            _, thet = integrated_pde_y(el, C1_solved, C2_solved)
            thetas.append(thet)
        theta = np.array(thetas)

    return theta

    # Define the temperature distribution function T(x)
def temperature_distribution_1(x, r, v):
    C1_var, C2_var  = sp.symbols('C1 C2', real=True)

    xw1 = x_tc[2]-r
    hv = 3.66 * k / (2 * r)
    Tfl = calculate_tfl(0, r, v)

    flux_0, t_0 = integrated_pde_x(0, C1_var, C2_var)
    flux_xw1, t_xw1 = integrated_pde_x(xw1, C1_var, C2_var)

    eq1= sp.Eq(flux_0, h*(Tw-t_0)) 
    eq2 = sp.Eq(flux_xw1, -hv*(t_xw1-Tfl))

    # Solve for C1 and C2
    solution = sp.solve([eq1, eq2], (C1_var, C2_var))
    C1_solved, C2_solved = solution[C1_var], solution[C2_var]

    array = []
    for el in x:
        _, theta = integrated_pde_x(el, C1_solved, C2_solved)
        array.append(theta)

    return np.array(array)


def temperature_distribution_2(x, r, v):
    C1_var, C2_var = sp.symbols('C1 C2', real=True)

    xw2 = x_tc[2]+r
    hv = 3.66 * k / (2 * r)
    Tfl = calculate_tfl(0, r, v)

    flux_xw2, t_xw2 = integrated_pde_x(xw2, C1_var, C2_var)
    _, t_L0 = integrated_pde_x(2*L0, C1_var, C2_var)

    eq1= sp.Eq(flux_xw2, hv*(t_xw2-Tfl)) 
    eq2 = sp.Eq(t_L0, Ttis)

    solution = sp.solve([eq1, eq2], (C1_var, C2_var))
    C1_solved, C2_solved = solution[C1_var], solution[C2_var]

    array = []
    for el in x:
        _, theta = integrated_pde_x(el, C1_solved, C2_solved)
        array.append(theta)

    return np.array(array)

def t_distribution(x_arr, r, v):
    xw1 = x_tc[2]-r
    xw2 = x_tc[2]+r
    Tfl = calculate_tfl(0, r, v)

    T = np.where(x_arr <= xw1, temperature_distribution_1(x_arr, r, v),
                    np.where((xw1 < x_arr) & (x_arr <= xw2), Tfl,
                            np.where((xw2 < x_arr) & (x_arr <= 2*L0), temperature_distribution_2(x_arr, r, v),
                                    Ttis
                                    )))
    return T

# Calculate step size
N = 100  # number of steps
dy = L / N
y = np.linspace(-L/2, L/2, N)

def calculate_tfl_distrib(v, R1):
    # Heat transfer coefficient (h)
    hv = 3.66 * k / (2 * R1)

    # Arrays to store temperature values
    T_fluid = np.zeros(N)
    T_w1 = np.zeros(N)
    T_w2 = np.zeros(N)
    Q_in = np.zeros(N)
    Q_out = np.zeros(N)

    # Set initial values
    T_fluid[0] = T_fluid_initial
    T_w1[0] = calculate_tw1(y[0], R1)
    T_w2[0] = calculate_tw2(y[0], R1)
    Q_in[0] = hv*(T_w1[0]-T_fluid[0])
    Q_out[0] = hv*(T_fluid[0]-T_w2[0])
    

    # Iterative computation
    for i in range(N - 1):
        # Update fluid temperature
        T_fluid[i + 1] = T_fluid[i] + (Q_in[i]-Q_out[i])/(rho*cfl*v)
        T_w1[i + 1] = calculate_tw1(y[i+1], R1)
        T_w2[i + 1] = calculate_tw2(y[i+1], R1)
        Q_in[i + 1] = hv*(T_w1[i + 1]-T_fluid[i + 1])
        Q_out[i + 1] = hv*(T_fluid[i + 1]-T_w2[i + 1])
    
    df = np.vstack((y, T_fluid, T_w1, T_w2, Q_in, Q_out))

    return df


# Generate depth values (x) and calculate temperature
depth = np.linspace(0, 3*L0, 100) 


eng = matlab.engine.start_matlab()
eng.cd(src_dir, nargout=0)
eng.simple_script(nargout=0)
eng.quit()
a = np.loadtxt(f"{src_dir}/output_pbhe.txt")
x_values_matlab = a[:, 0]*3*L0
pbhe1_matlab = a[:, 1]
pbhe2_matlab = a[:, 2]
pbhe3_matlab = a[:, 3]

file_path = f"{src_dir}/data/vessel/20240522_1.txt"  # Replace with your file path
timeseries_data = ivd.load_measurements(file_path)
meas1 = ivd.extract_entries(timeseries_data, 2*60, 3*60)
meas2 = ivd.extract_entries(timeseries_data, 30*60, 35*60)
meas3 = ivd.extract_entries(timeseries_data, 56*60, 58*60)
meas4 = ivd.extract_entries(timeseries_data, 83*60, 83*60+30)

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

temp_vessel1 = calculate_tfl_distrib(R1_vessel1, v_vessel1)
temp_vessel2 = calculate_tfl_distrib(R1_vessel2, v_vessel2)
temp_vessel3 = calculate_tfl_distrib(R1_vessel3, v_vessel3)

vessel1 = t_distribution(depth, R1_vessel1, v_vessel1)
vessel2 = t_distribution(depth, R1_vessel2, v_vessel2)
vessel3 = t_distribution(depth, R1_vessel3, v_vessel3)


# Plotting
plt.figure(figsize=(8, 6))
plt.plot(depth * 100, vessel1, label="Analytic Distribution 1")
plt.plot(depth * 100, vessel2, label="Analytic Distribution 2")
plt.plot(depth * 100, vessel3, label="Analytic Distribution 3")
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


# Plot results
plt.figure(figsize=(10, 5))

# Plot each vessel
plt.plot(y, temp_vessel1, label='Vessel 1')
plt.plot(y, temp_vessel2, label='Vessel 2')
plt.plot(y, temp_vessel3, label='Vessel 3')

# Directly label the lines
plt.text(100 * L - 5, temp_vessel1[-1] + 0.01, 'Vessel 1', fontsize=12, color='C0')
plt.text(100 * L - 5, temp_vessel2[-1] + 0.01, 'Vessel 2', fontsize=12, color='C1')
plt.text(100 * L - 5, temp_vessel3[-1] + 0.01, 'Vessel 3', fontsize=12, color='C2')

plt.xlabel(r'Location along the vessel, $y$ (cm)', fontsize=12)
# plt.ylabel(r'$\frac{T_{\text{fluid}}(x)-T_{\text{fluid}}(0)}{T_{\text{tumour}}-T_{\text{fluid}}(0)}$', fontsize=15)
plt.ylabel(r'$T_{\text{fluid}}(y)$', fontsize=15)
plt.title('Temperature as a Function of Location Along Vessels', fontsize=14)
plt.grid(True)
plt.savefig(f"{src_dir}/data/simulations/analytical_along_vessel.png", dpi=120)
plt.show()
