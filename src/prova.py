import numpy as np

W = 0.5
n_obs = 8
variation = 0.2

W_tot = np.linspace(0.5*(1-variation), 0.5*(1+variation), n_obs)

print(W_tot)