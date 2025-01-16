import os
from omegaconf import OmegaConf
import numpy as np
from hydra import initialize, compose


current_file: str = os.path.abspath(__file__)
src_dir: str = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
models: str = os.path.join(git_dir, "models")

conf_dir: str = os.path.join(src_dir, "configs")

initialize('configs', version_base=None)
# cfg = compose(config_name='config_run')
cfg = OmegaConf.load(f"{conf_dir}/config_run.yaml")

props = cfg.model_properties
pars = cfg.model_parameters
exp = cfg.experiment
meas_settings = getattr(cfg.experiment_type, exp.run)
props.Ty10, props.Ty20, props.Ty30 = meas_settings.Ty10, meas_settings.Ty20, meas_settings.Ty30

n_ins: int = props.n_ins
n_anchor_points: int = props.n_anchor_points

L0: float = props.L0
tauf: float = props.tauf
k: float = props.k
c: float = props.c
rho: float = props.rho
c_b: float = c
rho_b: float = rho
h: float = props.h

beta: float = props.beta
SAR_0: float = props.SAR_0
PD: float = props.PD
x0: float = props.x0
pwr_fact: float = props.pwr_fact

Tmax: float = props.Tmax
Troom: float = props.Troom
Ty10: float = props.Ty10
Ty20: float = props.Ty20
Tgt20: float = props.Tgt20
Ty30: float = props.Ty30
dT: float = (Tmax-Troom)

alfa: float = props.alfa
b1: float = props.b1
b2: float = props.b2
b3: float = props.b3
b4: float = props.b4

c1: float = props.c1
c2: float = props.c2
c3: float = props.c3

x_gt1: float = pars.x_gt1
x_gt2: float = pars.x_gt2

W0: float = pars.W0
W1: float = pars.W1
W2: float = pars.W2
W3: float = pars.W3
W4: float = pars.W4
W5: float = pars.W5
W6: float = pars.W6
W7: float = pars.W7
W_min: float = pars.W_min
W_max: float = pars.W_max
W_sys: float = pars.W_sys
W_index: int = pars.W_index
n_obs: int = pars.n_obs

obs = np.array([W0, W1, W2, W3, W4, W5, W6, W7])
W_obs = float(obs[W_index])

lamb: float = pars.lam  # Access the lambda parameter
upsilon: float = pars.upsilon

def rescale_t(t: float)->float:

    return Troom + t *(Tmax - Troom)

def scale_t(t: float)->float:

    return (t - Troom)/(Tmax - Troom)

"coefficients a1, a2, a3, a4, a5"

a1: float = round((L0**2/tauf)*((rho*c)/k), 7)
a2: float = round(L0**2*rho_b*c_b/k, 7)
cc: float = np.log(2)/(PD - 10**(-2)*x0)
a3: float = round(pwr_fact*rho*L0**2*beta*SAR_0*np.exp(cc*x0)/k*dT, 7)
a4: float = round(cc*L0, 7)
a5: float = round(L0*h/k, 7)
K: float = alfa*L0


eta: float = np.where(K>=(np.pi**2)/4, (np.pi**2)/4, K)

decay_rate_exact: float = (eta/a1+W_sys*a2/a1)

c_0: float = (np.abs(W_obs*a2/a1 - W_sys*a2/a1)**2)/(eta/a1 + W_obs*a2/a1)**2

decay_rate_diff: float = (eta/a1+W_obs*a2/a1)/2

theta10, theta20, theta30, theta_gt20 = scale_t(Ty10), scale_t(Ty20), scale_t(Ty30), scale_t(Tgt20)
X_gt2 = x_gt2/L0

# Define the equations in matrix form
A = np.array([
    [1, 1, 1, 1],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [X_gt2**3, X_gt2**2, X_gt2, 1]
])

B = np.array([theta10, theta20, -a5 * (theta30 - theta20), theta_gt20])

# Solve the system of equations
sol = np.linalg.solve(A, B)

# Extract the solutions
b1, b2, b3, b4 = [round(val, 5) for val in sol]

if n_ins>2:
    c3 = round(theta20, 5)
    c2 = round(-a5 * (theta30 - theta20), 5)
    c1 = round(theta10 - c2 - c3, 5)
else:
    c1, c2, c3 = None, None, None

props.b1, props.b2, props.b3, props.b4 = float(b1), float(b2), float(b3), float(b4)
props.c1, props.c2, props.c3 = float(c1), float(c2), float(c3)

OmegaConf.save(cfg, f"{conf_dir}/config_run.yaml")


if __name__ == "__main__":
    print(b1, b2, b3, b4, c1, c2, c3)