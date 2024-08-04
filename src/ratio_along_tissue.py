import numpy as np
import matplotlib.pyplot as plt
import ratio_along_vessel as ral

# Constants for Vessel 2
R1 = 1.0 / 1000  # radius in meters (1 mm)
R2 = 2.0 / 100  # radius in meters (2 cm)
L = 0.25  # length of the vessel in meters
v = 1.5 / 100  # velocity in m/s (1.5 cm/s)
k = 0.6  # conductivity of tissue in W/m°C
cfl = 4186  # specific heat capacity of fluid in J/kg°C
rho = 1000  # density of fluid in kg/m^3
T_tumour = 42  # tumour temperature in °C
T_blood_initial = 37  # initial blood temperature in °C

# Grid settings
Nx = 100  # number of points along the x-axis
Nr = 100  # number of points along the radial direction
x = np.linspace(0, L, Nx)  # x-axis (length along vessel)
r = np.linspace(0, 2*R1, Nr)  # radial axis (from center to surface)
X, R = np.meshgrid(x, r)  # create meshgrid for plotting

# Heat transfer coefficients and resistances
h = 3.66 * k / (2 * R1)  # heat transfer coefficient
R_w = 1 / (2 * np.pi * R1 * h)  # resistance of the vessel wall
R_t = np.log(R2 / R1) / (2 * np.pi * k)  # resistance of the tissue
R_f = 1 / (np.pi * (R1 ** 2) * v * cfl * rho)  # resistance of the fluid

Q, T_blood = ral.vessel_temperature_distribution(v, R1)

# Temperature distribution model
def temperature_distribution(x, r):
    T = T_blood + Q * (R_w + np.log(r/R1)/(2*np.pi*k*0.01))
    return T

# Calculate temperature for each point in the grid
T = temperature_distribution(X, R)

# Plot the 2D temperature distribution
plt.figure(figsize=(12, 6))
contour = plt.contourf(X*100, R*1000, T, levels=50, cmap='hot')
cbar = plt.colorbar(contour)
cbar.set_label('Temperature (°C)', fontsize=12)

plt.xlabel('Location along the vessel, $x$ (cm)', fontsize=12)
plt.ylabel('Radial distance from center, $r$ (mm)', fontsize=12)
plt.title('2D Temperature Distribution in Vessel 2 Plane', fontsize=14)
plt.show()