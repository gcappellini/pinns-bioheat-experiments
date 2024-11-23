import os
from omegaconf import OmegaConf
import numpy as np


current_file: str = os.path.abspath(__file__)
src_dir: str = os.path.dirname(current_file)

conf_dir: str = os.path.join(src_dir, "configs")
cfg: dict = OmegaConf.load(f"{src_dir}/config.yaml")


L0: float = cfg.model_properties.L0
tauf: float = cfg.model_properties.tauf
k: float = cfg.model_properties.k
c: float = cfg.model_properties.c
rho: float = cfg.model_properties.rho
h: float = cfg.model_properties.h

beta: float = cfg.model_properties.beta
SAR_0: float = cfg.model_properties.SAR_0
PD: float = cfg.model_properties.PD
x0: float = cfg.model_properties.x0
pwr_fact: float = cfg.model_properties.pwr_fact

Tmax: float = cfg.model_properties.Tmax
Troom: float = cfg.model_properties.Troom
Ty10: float = cfg.model_properties.Ty10
Ty20: float = cfg.model_properties.Ty20
Ty30: float = cfg.model_properties.Ty30
dT: float = (Tmax-Troom)

alfa: float = cfg.model_properties.alfa
b2: float = cfg.model_properties.b2

# Accessing model parameters from the config
W0: float = cfg.model_parameters.W0
W1: float = cfg.model_parameters.W1
W2: float = cfg.model_parameters.W2
W3: float = cfg.model_parameters.W3
W4: float = cfg.model_parameters.W4
W5: float = cfg.model_parameters.W5
W6: float = cfg.model_parameters.W6
W7: float = cfg.model_parameters.W7
W_sys: float = cfg.model_parameters.W_sys
W_obs: float = cfg.model_parameters.W_obs

lamb: float = cfg.model_parameters.lam  # Access the lambda parameter
upsilon: float = cfg.model_parameters.upsilon

def rescale_t(t: float)->float:

    return Troom + t *(Tmax - Troom)

"coefficients a1, a2, a3, a4, a5"

a1: float = round((L0**2/tauf)*((rho*c)/k), 7)
a2: float = round(L0**2*rho*c/k, 7)
cc: float = np.log(2)/(PD - 10**(-2)*x0)
# cc = 16
a3: float = round(pwr_fact*rho*L0**2*beta*SAR_0*np.exp(cc*x0)/k*dT, 7)
a4: float = round(cc*L0, 7)
a5: float = round(L0*h/k, 7)
K: float = alfa*L0


eta: float = np.where(K>=(np.pi**2)/4, (np.pi**2)/4, K)

decay_rate_exact: float = (eta/a1+W_sys*a2/a1)

c_0: float = (np.abs(W_obs*a2/a1 - W_sys*a2/a1)**2)/(eta/a1 + W_obs*a2/a1)**2

decay_rate_diff: float = (eta/a1+W_obs*a2/a1)/2
# print(f"Wsys:{W_sys}, Wobs:{W_obs}, c0: {c_0}, decay exact:{decay_rate_exact}, decay diff:{decay_rate_diff}")

# if __name__ == "__main__":


