import numpy as np
import utils

h_unk = np.linspace(8, 160, num=10).round(1)
# jj =  [1, 4, 20, 100, 500, 10000]

# errs = {}

# for ii in jj:
#     utils.set_K(ii)

#     multi_obs = {}  

#     for hh in h_unk:
#         modelu = utils.create_observer(hh)
#         modelu = utils.restore_model(modelu, f"obs_{hh}")
#         utils.test_observer(modelu, f"obs_{hh}")
#         multi_obs[hh] = modelu
#         errs[(utils.K, hh)] = utils.compute_l2(modelu)

# utils.plot_l2_vs_k(errs)

# lambdas = [10, 200, 1000]
# utils.mm_ode(multi_obs, lambdas)
# utils.plot_weights(lambdas)

utils.plot_mm_observer(h_unk, 20, 1000)