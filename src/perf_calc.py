import os
import numpy as np
from uncertainties import ufloat
import matplotlib.pyplot as plt

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

# Constants
L = 0.1  # length of the vessel in meters
N = 100  # number of steps
R1 = 1.0  # internal radius of the vessel
R2 = 1.4  # external radius of the vessel
k = 1.0 # conductivity of tissue

# R_IT = 0.5  # resistance of tissue
# R13 = 0.8  # some constant related to blood heating
h = 100  # heat transfer coefficient
# Ax = 0.001  # area of heat exchange
# M = 0.1  # mass flow rate of blood
cfl = 4186  # specific heat capacity of fluid in J/kg°C
rho = 1000  # density of fluid in kg/m^3
v = 1.0 # velocity of fluid
T_tumour = 42  # tumour temperature in °C



# Initial conditions
T_blood_initial = 37  # initial blood temperature in °C
T_wall_initial = 37  # initial wall temperature in °C

# Calculate step size
dx = L / N

# Resistance of vessel wall
R_w = 1/(2*np.pi*R1*h*dx)

# Resistance of tissue
R_t = np.log(R2/R1)/(2*np.pi*k*dx)

# Resistance of fluid
R_f = 1/(np.pi*(R1**2)*v*cfl*rho)

# Arrays to store temperature values
T_blood = np.zeros(N)
T_wall = np.zeros(N)

# Set initial values
T_blood[0] = T_blood_initial
T_wall[0] = T_wall_initial

# Iterative computation
for i in range(N - 1):
    # Heat flow through the vessel wall
    Q_wall = (T_wall[i] - T_blood[i])/R_w
    
    # Heat flow through the tissue
    Q_tissue = (T_tumour - T_wall[i])/R_t
    
    # # Fluid heating
    # Q_fluid = (T_blood[i+1] - T_blood[i])/R_f
    
    # Update fluid temperature
    T_blood[i+1] = T_blood[i] + (R_f/R_t)*(T_tumour-((R_w*R_t)/(R_w + R_t)*((T_blood[i]/R_w)+(T_tumour/R_t))))
    
    # # Update wall temperature
    # T_wall[i+1] = T_wall[i] + Q_wall * dx / R1

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(np.linspace(0, L, N), T_blood, label='Blood Temperature')
plt.plot(np.linspace(0, L, N), T_wall, label='Wall Temperature')
plt.xlabel('Position along the vessel (m)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Distribution in Blood and Vessel Wall')
plt.legend()
plt.grid(True)
plt.show()

