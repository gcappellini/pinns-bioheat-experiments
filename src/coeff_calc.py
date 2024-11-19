import os
from omegaconf import OmegaConf
import numpy as np


current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

conf_dir = os.path.join(src_dir, "configs")
cfg = OmegaConf.load(f"{src_dir}/config.yaml")

L0 = cfg.model_properties.L0
tauf = cfg.model_properties.tauf
k = cfg.model_properties.k
c = cfg.model_properties.c
rho = cfg.model_properties.rho
h = cfg.model_properties.h

beta = cfg.model_properties.beta
SAR_0 = cfg.model_properties.SAR_0
PD = cfg.model_properties.PD
x0 = cfg.model_properties.x0
pwr_fact = cfg.model_properties.pwr_fact

Tmax = cfg.model_properties.Tmax
Troom = cfg.model_properties.Troom
Ty10 = cfg.model_properties.Ty10
Ty20 = cfg.model_properties.Ty20
Ty30 = cfg.model_properties.Ty30
dT = (Tmax-Troom)

alfa = cfg.model_properties.alfa
b2 = cfg.model_properties.b2

# Accessing model parameters from the config
W0 = cfg.model_parameters.W0
W1 = cfg.model_parameters.W1
W2 = cfg.model_parameters.W2
W3 = cfg.model_parameters.W3
W4 = cfg.model_parameters.W4
W5 = cfg.model_parameters.W5
W6 = cfg.model_parameters.W6
W7 = cfg.model_parameters.W7
W_sys = cfg.model_parameters.W_sys
W_obs = cfg.model_parameters.W_obs

lamb = cfg.model_parameters.lam  # Access the lambda parameter
upsilon = cfg.model_parameters.upsilon

def rescale_t(t):

    return Troom + t *(Tmax - Troom)

"coefficients a1, a2, a3, a4, a5"

a1 = round((L0**2/tauf)*((rho*c)/k), 7)
a2 = round(L0**2*rho*c/k, 7)
cc = np.log(2)/(PD - 10**(-2)*x0)
# cc = 16
a3 = round(pwr_fact*rho*L0**2*beta*SAR_0*np.exp(cc*x0)/k*dT, 7)
a4 = round(cc*L0, 7)
a5 = round(L0*h/k, 7)
K = alfa*L0


eta = np.where(K>=(np.pi**2)/4, (np.pi**2)/4, K)

decay_rate_exact= (eta/a1+W_sys*a2/a1)

c_0 = (np.abs(W_obs*a2/a1 - W_sys*a2/a1)**2)/(eta/a1 + W_obs*a2/a1)**2

decay_rate_diff= (eta/a1+W_obs*a2/a1)/2
# print(f"Wsys:{W_sys}, Wobs:{W_obs}, c0: {c_0}, decay exact:{decay_rate_exact}, decay diff:{decay_rate_diff}")


