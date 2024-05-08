import numpy as np
import os

current_file = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file)
output_dir = f"{script_directory}/vessel"
os.makedirs(output_dir, exist_ok=True)

r1 = 0              # Vessel radius (m)
r2 = 0              # Tissue radius (m)
x = 0               # Position along the vessel (m)
Tb = 22             # Basal temperature (°C)
dT = 8              # Desired temperature increment (°C)
Tbolus = Tb + 4     # Bolus temperature (°C)

k = 0.6             # Thermal conductivity of wallpaper phantom (W/m°C)
Kb = 0.52           # Thermal conductivity of blood (W/m°C) https://itis.swiss/virtual-population/tissue-properties/database/thermal-conductivity/
h = 5000            # Heat transfer coefficient bolus-plastic (W/m2°C)
xpl = 1.5e-3        # Thickness of plastic bolus (m)
kpl = 0.2           # Thermal conductivity of plastic (W/m°C)

U = 1/((1/h)+(xpl/kpl))     # Convection coefficient bolus-phantom (W/m2°C)

# If there are no discontinuities in the temperature distribution the heat transfer coefficient hb is given
# for a laminar flow vessel by (Drew et a1 1936a, Lagendijk 1982a):

hb = 3.66*Kb/2*r1 

def Tmix(a):
    # Mixing cup temperature
    return Tb

def Twall(a):
    # Temperature at the vessel wall
    return Tb + dT

def Q_sq(a):
    # Heat flux through the vessel wall (W/m2)
    return hb * (Twall(a) - Tmix(a))


# Easiest way to take into account the heating system:
# Temperature at r2 must be kept at a fixed value (must change for the chosen r1)
# This can be the tumor temperature which is kept constant by a heating system (microwave, ultrasound, etc.)

# integrale di Q_sq mi da heat flow through vessel point up to point a

# integrale di eq.9 mi da heat flow through the tissue over the same length

# heating of the blood at a point x is given by

# queste tre equazioni devono essere uguali allo steady state. Devo aggiungere anche heat removal del water bolus

# Temperature gradient at the point x in tissue is given by eq. 8



