import os
from omegaconf import OmegaConf


current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

cfg = OmegaConf.load(f'{src_dir}/config.yaml')

L0 = cfg.model_properties.L0
tauf = cfg.model_properties.tauf
k = cfg.model_properties.k
c = cfg.model_properties.c
rho = cfg.model_properties.rho
h = cfg.model_properties.h

Tmax = cfg.model_properties.Tmax
Troom = cfg.model_properties.Troom

K = cfg.model_properties.K
delta = cfg.model_properties.delta

# Accessing model parameters from the config
W0 = cfg.model_parameters.W0
W1 = cfg.model_parameters.W1
W2 = cfg.model_parameters.W2
W3 = cfg.model_parameters.W3
W4 = cfg.model_parameters.W4
W5 = cfg.model_parameters.W5
W6 = cfg.model_parameters.W6
W7 = cfg.model_parameters.W7

lamb = cfg.model_parameters.lam  # Access the lambda parameter
upsilon = cfg.model_parameters.upsilon

def scale_t(t):

    return (t - Troom) / (Tmax - Troom)

"coefficients a1, a2, a3, a4"

a1 = round((L0**2/tauf)*((rho*c)/k), 7)
a2 = round(L0**2*c/k, 7)
a3 = round(L0*h/k, 7)





