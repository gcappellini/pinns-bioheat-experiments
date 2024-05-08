import numpy as np
import os
import matplotlib.pyplot as plt

current_file = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file)
output_dir = f"{script_directory}/vessel"
os.makedirs(output_dir, exist_ok=True)

r1 = 0.45/1000       # Vessel radius (m)
r2 = 0.1            # Tissue radius (m)

Tb = 22             # Basal temperature (°C)
dT = 8              # Desired temperature increment (°C)
Tbolus = Tb + 4     # Bolus temperature (°C)

k = 0.6             # Thermal conductivity of water (W/m°C)
kpl = 0.2           # Thermal conductivity of plastic (W/m°C)
keff = 0.6          # Effective thermal conductivity for wallpaper-paste filled phantom (W/m°C)

rho_w = 997         # Density of water (kg/m3)
cp_w = 4184         # Specific heat of water (J/kg°C)

# If there are no discontinuities in the temperature distribution the heat transfer coefficient hb is given
# for a laminar flow vessel by (Drew et a1 1936a, Lagendijk 1982a):

hb = 3.66*k/2*r1 
v_flow = 8/100                  # Flow velocity (m/s)
x = np.linspace(0, 0.5, 100)    # Axial coordinate of the vessel (m) 

# From Creeze 1992, Appendix 3
def xeq(v):
    return (1/2) * rho_w * cp_w * v * r1**2 * (2/4.01 + (k/keff) * np.log(r2/r1)) / k

def Tv(x, v):
    # return Tb * (1 - np.exp(-x/xeq(v)))
    return Tb*(1-np.exp(-x/xeq(v)))+Tb

print("Equilibrium lenght is", xeq(v_flow), "m")

# Plot Tv versus r1 and v
plt.figure(figsize=(10, 6))
plt.plot(x, Tv(x, v_flow))
plt.xlabel('vessel lenght (m)')
plt.ylabel('mixed cup temperature')
plt.title('Tv versus x')
plt.grid(True)
plt.savefig(f'{output_dir}/mixed_cup_temperature.png')
plt.show()
plt.clf()
