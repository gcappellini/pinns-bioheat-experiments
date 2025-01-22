import os
import numpy as np
from omegaconf import OmegaConf
import hydra
# import common as co
import plots as pp
import coeff_calc as cc
from scipy import integrate
import utils as uu
import time
import torch.nn as nn
import deepxde as dde
import torch
from common import set_run, generate_config_hash
# import matlab.engine
import utils as uu 
import common as co

np.random.seed(237)

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
conf_dir = os.path.join(src_dir, "configs")
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
models = os.path.join(git_dir, "models")
os.makedirs(tests_dir, exist_ok=True)


config = OmegaConf.load(f"{conf_dir}/config_run.yaml")

config.experiment.run="meas_cool_1"

rho = config.model_properties.rho   # Density (kg/m^3)
props = config.model_properties
pars = config.model_parameters

exp = getattr(config.experiment_type, "meas_cool_1")
# print(exp["Tgt20"])

r_1 = 0.5/1000
h_ves = 3.66*props.k/(2*r_1)
Q2 = (props.k/(props.L0-pars.x_gt2))*(exp["Tgt20"]-exp["Ty10"])
Tfl=props.Troom
Twall = Tfl + Q2/h_ves
# print(Twall)

coords = [0, pars.x_gt2, pars.x_w, pars.x_gt1, props.L0]
L=0.4
N = 100
# Calculate step size
dy = L / N
y = np.linspace(-L/2, L/2, N)
R2 = coords[3]-coords[2]
cfl=props.c

r_cooling_1 = 0.5/1000  # Radius Cooling 1 (m)
r_cooling_2 = 1.0/1000  # Radius Cooling 2 (m)

Q_cooling_1 = 0.4/17*(1/1000)*(1/60)    # Volumetric flow Cooling 1 (m^3/s)
Q_cooling_2 = 1.9/17*(1/1000)*(1/60)    # Volumetric flow Cooling 2 (m^3/s)

A_cooling_1 = np.pi*(r_cooling_1**2)    # Area Cooling 1 (m^2)
A_cooling_2 = np.pi*(r_cooling_2**2)    # Area Cooling 2 (m^2)

v_cooling_1 = round(Q_cooling_1/A_cooling_1, 3) # Velocity Cooling 1 (m/s)
v_cooling_2 = round(Q_cooling_2/A_cooling_1, 3)/2 # Velocity Cooling 2 (m/s)

print(v_cooling_1, v_cooling_2)

# def Tgt2(y):
    
#     y_normalized = y / (L / 2)
#     res = exp["Tgt20"] - (exp["Tgt20"] - props.Troom) * (y_normalized ** 2)
#     return res

# def calculate_keff(w):
#     return props.k*(1 + (0.2/w)*w)  # alpha = (keff/k -1)/w,        From Lagendijk chapter pg.27: keff=7.2, k=6.0


# # Function to calculate temperature distribution for a vessel
# def vessel_temperature_distribution(v, R1, eta=1.0):
#     # Heat transfer coefficient (h)
#     keff = props.k
#     keff = calculate_keff(pars.W_max)
#     h = 3.66 * keff / (2 * R1)

#     # Resistances
#     R_w = 1 / (2 * np.pi * R1 * h * dy)
#     R_p = np.log(R2 / R1) / (2 * np.pi * keff * dy)
#     R_f = 1 / (np.pi * (R1 ** 2) * v * cfl * rho)

#     # Arrays to store temperature values
#     T_fluid = np.zeros(N)
#     Q = np.zeros(N)

#     # Set initial values
#     T_fluid[0] = props.Troom
#     Q[0] = eta*(Tgt2(y[0])-T_fluid[0])/(R_f+R_w+R_p)

#     # Iterative computation
#     for i in range(N - 1):
#         # Update fluid temperature
#         T_fluid[i + 1] = T_fluid[i] + Q[i]*R_f
#         Q[i+1] = (Tgt2(y[i+1])-T_fluid[i + 1])/(R_f+R_w+R_p)
    
#     # df = pd.DataFrame({'x':x, 't_fluid':T_fluid, 't_wall':T_wall, 'Q': Q})
#     df = np.vstack((y, T_fluid, Q))

#     # compute thermal equilibrium length

#     return df

# dd1 = vessel_temperature_distribution(v_cooling_1, r_cooling_1)
# T_fluid_cooling_1 = dd1[1]

# dd2 = vessel_temperature_distribution(v_cooling_2, r_cooling_2)
# T_fluid_cooling_2 = dd2[1]

# pp.plot_generic(
#     x=[y, y],
#     y=[T_fluid_cooling_1, T_fluid_cooling_2],
#     title='Temperature of the fluid along the vessel',
#     xlabel='y-Axis',
#     ylabel='Temperature',
#     legend_labels=["Cooling 1", "Cooling 2"],
#     filename=f"{tests_dir}/plot_t_fluid_new.png",
#     colors=["olive", "cyan"]
# )

# # pp.plot_generic(
# #     x=[y],
# #     y=[Tgt2(y)],
# #     title='Tgt2 along the vessel',
# #     xlabel='y-Axis',
# #     ylabel='Temperature',
# #     legend_labels=["Tgt2"],
# #     filename=f"{tests_dir}/plot_tgt2.png",
# #     colors=["olive"]
# # )

# res_cool_1 = float(round(T_fluid_cooling_1[50], 2))
# res_cool_2 = float(round(T_fluid_cooling_2[50], 2))
# print(res_cool_1,res_cool_2 )

# exp.Tfl=res_cool_1
# exp2 = getattr(config.experiment_type, "meas_cool_2")
# exp2.Tfl=res_cool_2

# # OmegaConf.save(config, f"{conf_dir}/config_run.yaml")



# # Tgt20 = exp.Tgt20
# # Tfl = exp.Tfl
# # x = np.linspace(0, 1, num=100)
# # scaled_r = r_cooling_1/cc.L0
# # theta_w=0.7
# # theta_pred = np.cos(x)
# # theta_fluid = 0.4

# # meas_sett = getattr(config.experiment_type, config.experiment.run)
# # T_cyl_arr = uu.filter_theta_vessel(np.vstack((x,x)).T, meas_sett["r1"], meas_sett["Tfl"], theta_pred)

# # pp.plot_generic(
# #     x=[x],
# #     y=[T_cyl_arr],
# #     title='T along the phantom',
# #     xlabel='x-Axis',
# #     ylabel='Temperature',
# #     legend_labels=["T_cyl_arr"],
# #     filename=f"{tests_dir}/plot_t_cyl_arr.png",
# #     colors=["olive"]
# # )