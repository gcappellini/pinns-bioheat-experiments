import numpy as np
import matplotlib.pyplot as plt
import os
import json
import import_vessel_data as ivd
import sympy as sp
from sympy.solvers import solve
from scipy import special
import simulation as se


current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

with open(f"{src_dir}/properties.json", 'r') as f:
  data = json.load(f)

# Constants
L, L0 = data["L"], data["L0"]
k, cfl, rho, h = data["k"], data["c"], data["rho"], data["h"]

Tmax, Ttis, Tw, dT= data["Tmax"], data["Ttis"], data["Tw"], data["dT"] # tumour temperature in °C

P0, d, a, b, x0, beta = data["P0"], data["d"], data["a"], data["b"], data["x0"], data["beta"]
P0_x = beta * P0 * np.exp(a*x0)

W1, W2, W3, alpha, eta = data["W1"], data["W2"], data["W3"], data["alpha"], data["eta"]


x_tc = np.array([0., 0.015, 0.026, 0.048, 0.082])
R2 = 0.010
xr2 = x_tc[2]-R2

N = 100
# Calculate step size
dy = L / N
y = np.linspace(-L/2, L/2, N)

def calculate_keff(w):
    return k*(1 + alpha*w)


def integrated_pde_x(x, C1, C2, w):
    flux = P0_x * np.exp(-b*((L**2)/4))* np.exp(-a*x)+C1
    temperature = (1/calculate_keff(w))*(-a*P0_x*np.exp(-b*((L**2)/4))*np.exp(-a*x)+C1*x+C2)
    return flux, temperature

def Tr2_start(w):
    C1_var, C2_var  = sp.symbols('C1 C2', real=True)

    flux_0, t_0 = integrated_pde_x(0, C1_var, C2_var, w)
    _, t_L0 = integrated_pde_x(L0, C1_var, C2_var, w)

    eq1= sp.Eq(flux_0, (h/calculate_keff(w))*(Tw-t_0)) 
    eq2 = sp.Eq(t_L0, Ttis)

    # Solve for C1 and C2
    solution = sp.solve([eq1, eq2], (C1_var, C2_var))
    C1_solved, C2_solved = solution[C1_var], solution[C2_var]

    # if isinstance(x, int):
    _, theta = integrated_pde_x(-L/2, C1_solved, C2_solved, w)


    return theta


def Tr2(y, w):
    f = se.temperature_distribution(xr2, w)
    tmin = Ttis
    e1 = 4*(tmin-f)/(L**2)

    return e1*y**2 + f

# Function to calculate temperature distribution for a vessel
def vessel_temperature_distribution(v, R1, w):
    # Heat transfer coefficient (h)
    keff = calculate_keff(w)
    h = 3.66 * keff / (2 * R1)

    # Resistances
    R_w = 1 / (2 * np.pi * R1 * h * dy)
    R_t = np.log(R2 / R1) / (2 * np.pi * keff * dy)
    R_f = 1 / (np.pi * (R1 ** 2) * v * cfl * rho)

    # Arrays to store temperature values
    T_fluid = np.zeros(N)
    Q = np.zeros(N)

    # Set initial values
    T_fluid[0] = Ttis
    Q[0] = eta*(Tr2(y[0], w)-T_fluid[0])/(R_f+R_w+R_t)

    # Iterative computation
    for i in range(N - 1):
        # Update fluid temperature
        T_fluid[i + 1] = T_fluid[i] + Q[i]*R_f
        Q[i+1] = (Tr2(y[i+1], w)-T_fluid[i + 1])/(R_f+R_w+R_t)
    
    # df = pd.DataFrame({'x':x, 't_fluid':T_fluid, 't_wall':T_wall, 'Q': Q})
    df = np.vstack((y, T_fluid, Q))

    return df

def scale_t(t):
    return (t - Ttis) / (Tmax - Ttis)



# Vessel 1 Parameters
R1_vessel1 = 0.5 / 1000  # radius in meters
v_vessel1 = 1.0 / 100  # velocity in m/s

# Vessel 2 Parameters
R1_vessel2 = 1.0 / 1000  # radius in meters
v_vessel2 = 1.5 / 100  # velocity in m/s

# Vessel 3 Parameters
R1_vessel3 = 1.5 / 1000  # radius in meters
v_vessel3 = 2.0 / 100  # velocity in m/s

# Calculate temperature distribution for each vessel
dd1 = vessel_temperature_distribution(v_vessel1, R1_vessel1, W1)
dd2 = vessel_temperature_distribution(v_vessel2, R1_vessel2, W2)
dd3 = vessel_temperature_distribution(v_vessel3, R1_vessel3, W3)

temp_vessel1, temp_vessel2, temp_vessel3 = dd1[1], dd2[1], dd3[1]
np.savez(f"{src_dir}/data/simulations/simulation/y_axis_{R1_vessel1}_{v_vessel1}_{W1}.npz", vessel=temp_vessel1)
np.savez(f"{src_dir}/data/simulations/simulation/y_axis_{R1_vessel2}_{v_vessel2}_{W2}.npz", vessel=temp_vessel2)
np.savez(f"{src_dir}/data/simulations/simulation/y_axis_{R1_vessel3}_{v_vessel3}_{W3}.npz", vessel=temp_vessel3)

temp_ratio_vessel1 = scale_t(temp_vessel1)
temp_ratio_vessel2 = scale_t(temp_vessel2)
temp_ratio_vessel3 = scale_t(temp_vessel3)


# Plot results
plt.figure(figsize=(10, 5))
x_values = np.linspace(0, 100 * L, N)

# Plot each vessel
plt.plot(x_values, temp_ratio_vessel1, label='Vessel 1')
plt.plot(x_values, temp_ratio_vessel2, label='Vessel 2')
plt.plot(x_values, temp_ratio_vessel3, label='Vessel 3')

# Directly label the lines
plt.text(100 * L - 5, temp_ratio_vessel1[-1] + 0.01, 'Vessel 1', fontsize=12, color='C0')
plt.text(100 * L - 3, temp_ratio_vessel2[-1] + 0.08, 'Vessel 2', fontsize=12, color='C1')
plt.text(100 * L - 2, temp_ratio_vessel3[-1] + 0.03, 'Vessel 3', fontsize=12, color='C2')

plt.xlabel(r'Location along the vessel, $y$ (cm)', fontsize=12)
plt.ylabel(r'$\frac{T_{\text{fluid}}(y)-T_{\text{room}}}{T_{\text{max}}-T_{\text{room}}}$', fontsize=15)
plt.title('Temperature Ratio as a Function of Location Along Vessels', fontsize=14)
plt.grid(True)
# plt.savefig(f"{src_dir}/data/simulations/t_ratio_along_vessel.png", dpi=120)
plt.show()

plt.plot(y, Tr2(y, W1), label=f'W1 = {W1}')
plt.plot(y, Tr2(y, W2), label=f'W2 = {W2}')
plt.plot(y, Tr2(y, W3), label=f'W3 = {W3}')

plt.xlabel('y')
plt.ylabel(r'$T_{r2}(y, w)$')
plt.legend()
plt.title(r'$T_{r2}(y, w)$ vs $y$ for different values of $w$', fontsize=15)
plt.show()

