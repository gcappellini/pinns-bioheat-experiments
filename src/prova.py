import utils as uu
import os
import matlab.engine
import numpy as np
from scipy import integrate
import common as co
import plots as pp

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

# Ground truth with Matlab:
eng = matlab.engine.start_matlab()
eng.cd(src_dir, nargout=0)
eng.BioHeat(nargout=0)
eng.quit()

# PINNs implementation
co.set_prj("test_matlab")
run_figs = co.set_run("21")

multi_obs = uu.mm_observer()

X, y_sys, y_obs, y_mmobs = uu.gen_testdata()
x_obs = uu.gen_obsdata()

print(len(multi_obs), y_obs.shape)

for el in range(len(multi_obs)):
    pred = multi_obs[el].predict(x_obs)
    true = y_obs[:, el].reshape(pred.shape)
    pp.check_obs(X, true, pred, el, run_figs)
    pp.plot_l2_tf(X, true, pred, multi_obs[el], el, run_figs)


# Comparing observation error 
X, _, _, _ = uu.gen_testdata()
t = X[:, 1]
mu = uu.mu(multi_obs, t)
pp.plot_mu(mu, t, run_figs, gt=True)


# uu.plot_comparison(X, y_sys, y_mmobs)
# t, weights = uu.load_weights()
# uu.plot_weights(weights, t, gt=True)



# n_obs = 8

# x = uu.gen_obsdata()

# p0 = np.full((n_obs,), 1/n_obs)
# par = uu.read_json("parameters.json")
# lam = par["lambda"]

# def f(t, p):
#     a = uu.mu(multi_obs, t)
#     e = np.exp(-1*a)
#     d = np.inner(p, e)
#     f = []
#     for el in range(len(p)):
#         ccc = - lam * (1-(e[el]/d))*p[el]
#         f.append(ccc)
#     return np.array(f)


# sol = integrate.solve_ivp(f, (0, 1), p0, t_eval=np.linspace(0, 1, 100))
# x = sol.y
# t = sol.t
# weights = np.zeros((sol.y.shape[0]+1, sol.y.shape[1]))
# weights[0] = sol.t
# weights[1:] = sol.y
# np.save(f'{prj_logs}/weights_lam_{lam}.npy', weights)
# plot_weights(x, t, lam)
# plot_mu(multi_obs, t)
# metrics = mm_plot_and_metrics(multi_obs, lam, n_test)

# wandb.log(metrics)
# wandb.finish()

# a = uu.import_testdata()
# e = a[:, 0:2]
# theta_true = a[:, 2]
# g = uu.import_obsdata()
# uu.plot_comparison(e, )

# print(a)


