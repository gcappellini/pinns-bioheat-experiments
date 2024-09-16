import utils as uu
import os
import matlab.engine
import numpy as np
from scipy import integrate
import common as co
import wandb
import plots as pp

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

# Ground truth with Matlab:
eng = matlab.engine.start_matlab()
eng.cd(src_dir, nargout=0)
eng.BioHeat(nargout=0)
eng.quit()

# PINNs implementation
prj_figs = co.set_prj("test_matlab4")

# Comparing observation error 
X, y_sys, _, y_mmobs = uu.gen_testdata()
t = np.unique(X[:, 1])
# mu = uu.mu(multi_obs, t)
mu = uu.compute_mu()

pp.plot_mu(mu, t, prj_figs, gt=True)
pp.plot_comparison(X, y_sys, y_mmobs, prj_figs)
t, weights = uu.load_weights()
pp.plot_weights(weights, t, prj_figs, gt=True)

multi_obs = uu.mm_observer()

X, y_sys, y_obs, y_mmobs = uu.gen_testdata()
x_obs = uu.gen_obsdata()

for el in range(len(multi_obs)):
    run_figs = co.set_run(f"obs_{el}")
    aa = uu.read_json(f"{run_figs}/properties.json")
    wandb.init(
    project="mm_obs_amc_cooling", name=f"new_obs_{el}",
    config=aa
    )
    pred = multi_obs[el].predict(x_obs)
    y_sys = y_sys.reshape(pred.shape)
    pp.check_obs(X, y_obs[:, el], pred, el, run_figs)
    pp.plot_l2_tf(X, y_sys, pred, multi_obs[el], el, run_figs)
    b = uu.compute_metrics(y_obs[:, el], pred)
    wandb.log(b)
    wandb.finish()

t = np.unique(X[:, 1:2])
mu = uu.mu(multi_obs, t)
pp.plot_mu(mu, t, prj_figs)

n_obs = 8

x = uu.gen_obsdata()

p0 = np.full((n_obs,), 1/n_obs)
par = uu.read_json(f"{src_dir}/parameters.json")
lam = par["lambda"]

def f(t, p):
    a = uu.mu(multi_obs, t)
    e = np.exp(-1*a)
    d = np.inner(p, e)
    f = []
    for el in range(len(p)):
        ccc = - lam * (1-(e[:, el]/d))*p[el]
        f.append(ccc)
    return np.array(f).reshape(len(f),)


sol = integrate.solve_ivp(f, (0, 1), p0, t_eval=np.linspace(0, 1, 100))
x = sol.y
t = sol.t
weights = np.zeros((sol.y.shape[0]+1, sol.y.shape[1]))
weights[0] = sol.t
weights[1:] = sol.y
np.save(f'{prj_figs}/weights_lam_{lam}.npy', weights)
pp.plot_weights(weights[1:], weights[0], prj_figs)




