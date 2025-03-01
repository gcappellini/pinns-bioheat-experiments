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
cfg = compose(config_name='config_run')
# cfg = OmegaConf.load(f"{conf_dir}/config_run.yaml")

props = cfg.model_properties
pars = cfg.model_parameters
exp = cfg.experiment
meas_settings = getattr(cfg.experiment_type, exp.run)
props.Ty10, props.Ty20, props.Ty30 = meas_settings.Ty10, meas_settings.Ty20, meas_settings.Ty30

nins: int = props.nins
nanc: int = props.nanc
seed: int = props.seed

L0: float = props.L0
tf: float = props.tf
k: float = props.k
c: float = props.c
rho: float = props.rho
cb: float = c
rhob: float = rho
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
Tgt0: float = props.Tgt0
Ty30: float = props.Ty30
dT: float = (Tmax-Troom)

oig: float = props.oig
b1: float = props.b1
b2: float = props.b2
b3: float = props.b3
b4: float = props.b4

c1: float = props.c1
c2: float = props.c2
c3: float = props.c3

xgt1: float = pars.xgt1
xw: float = pars.xw
xgt: float = pars.xgt
Xgt = round(float(xgt / L0), 5)
Xw = round(float(xw / L0), 5)
Xgt1 = round(float(xgt1 / L0), 5)

pars.Xgt = Xgt
pars.Xw = Xw
pars.Xgt1 = Xgt1

wbmin: float = pars.wbmin
wbmax: float = pars.wbmax
wbsys: float = pars.wbsys
wbindex: int = pars.wbindex
nobs: int = pars.nobs

obs_steps = 8 if nobs<=8 else nobs
obs = np.logspace(np.log10(wbmin), np.log10(wbmax), obs_steps).round(6)
wbobs = float(obs[wbindex])
pars.wbobs = wbobs

eight_obs = np.logspace(np.log10(wbmin), np.log10(wbmax), 8).round(6)
for i in range(8):
    setattr(pars, f'wb{i}', float(eight_obs[i]))

lamb: float = pars.lam  # Access the lambda parameter
upsilon: float = pars.upsilon

def rescale_t(t: float)->float:

    return Troom + t *(Tmax - Troom)

def scale_t(t: float)->float:

    return round((t - Troom)/(Tmax - Troom), 5)

"coefficients a1, a2, a3, a4, a5"

a1: float = round((L0**2/tf)*((rho*c)/k), 7)
a2: float = round(L0**2*rhob*cb/k, 7)
cc: float = round(np.log(2)/(PD - 10**(-2)*x0), 7)
a3: float = round(pwr_fact*rho*L0**2*beta*SAR_0*np.exp(cc*x0)/k*dT, 7)
a4: float = round(cc*L0, 7)
a5: float = round(L0*h/k, 7)
props.a1 = float(a1)
props.a2 = float(a2)
props.a3 = float(a3)
props.a4 = float(a4)
props.a5 = float(a5)
props.cc = float(cc)

pwic: float = np.where(oig>=(np.pi**2)/4, (np.pi**2)/4, oig)

decay_rate_exact: float = (pwic/a1+wbsys*a2/a1)

c0: float = (np.abs(wbobs*a2/a1 - wbsys*a2/a1)**2)/(pwic/a1 + wbobs*a2/a1)**2

decay_rate_diff: float = (pwic/a1+wbobs*a2/a1)/2

theta10, theta20, theta30, thetagt0 = scale_t(Ty10), scale_t(Ty20), scale_t(Ty30), scale_t(Tgt0)

props.theta10 = float(theta10)
props.theta20 = float(theta20)
props.theta30 = float(theta30)
props.thetagt0 = float(thetagt0)


# Define the equations in matrix form
A = np.array([
    [1, 1, 1, 1],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [xgt**3, xgt**2, xgt, 1]
])

B = np.array([theta10, theta20, -a5 * (theta30 - theta20), thetagt0])

# Solve the system of equations
sol = np.linalg.solve(A, B)

# Extract the solutions
b1, b2, b3, b4 = [round(val, 5) for val in sol]

if nins>2:
    c3 = round(theta20, 5)
    c2 = round(-a5 * (theta30 - theta20), 5)
    c1 = round(theta10 - c2 - c3, 5)
    props.c1, props.c2, props.c3 = float(c1), float(c2), float(c3)
else:
    props.c1, props.c2, props.c3 = None, None, None

props.b1, props.b2, props.b3, props.b4 = float(b1), float(b2), float(b3), float(b4)


OmegaConf.save(cfg, f"{conf_dir}/config_run.yaml")


r_cat = 0.5/1000  # Radius Cooling 1 (m)
n_cat_cooling_1 = 1.0           # n° of catethers Cooling 1
n_cat_cooling_2 = 2.0           # n° of catethers Cooling 2

Q_cooling_1_lmin = 0.4/17                    # Volumetric flow Cooling 1 (L/min)
Q_cooling_2_lmin = 1.9/17                    # Volumetric flow Cooling 2 (L/min)

Q_cooling_1 = (0.4/17)*(1/1000)*(1/60)    # Volumetric flow Cooling 1 (m^3/s)
Q_cooling_2 = (1.9/17)*(1/1000)*(1/60)    # Volumetric flow Cooling 2 (m^3/s)

A_cooling_1 = np.pi*n_cat_cooling_1*(r_cat**2)    # Area Cooling 1 (m^2)
A_cooling_2 = np.pi*n_cat_cooling_2*(r_cat**2)    # Area Cooling 2 (m^2)

v_cooling_1 = round(Q_cooling_1/A_cooling_1, 3) # Velocity Cooling 1 (m/s)
v_cooling_2 = round(Q_cooling_2/A_cooling_2, 3) # Velocity Cooling 2 (m/s)

if __name__ == "__main__":
    print(os.path.abspath(cfg.run.dir))
    # print(f"{a1} & {a2} & {a3} & {a4} & {a5}")
    # print(f"{theta10} & {theta20} & {theta30} & {theta_gt20}")
    # print(f"{b1} & {b2} & {b3} & {b4}")
    # print(f"{c1} & {c2} & {c3} & {upsilon} & {lamb}")
    print(f"{Q_cooling_1} & {v_cooling_1}")
    print(f"{Q_cooling_2} & {v_cooling_2}")