import numpy as np
import utils

gains =  [1, 4, 20, 100, 500, 10000]
U = 369

for ii in gains:
    utils.set_K(ii)
    errs = {}
    modelu = utils.create_observer(U)
    modelu = utils.restore_model(modelu, f"obs_{U}")
    # utils.test_observer(modelu, f"obs_{U}")
    errs[(ii)] = utils.compute_l2(modelu)

utils.plot_l2_vs_k(errs)

# h_unk = np.linspace(8, 160, num=10).round(1)
# gains =  [1, 4, 20, 100, 500, 10000]
# lambdas = [10, 200, 1000]


# for ii in gains:
#     utils.set_K(ii)
#     errs = {}
#     multi_obs = {}  

#     for hh in h_unk:
#         modelu = utils.create_observer(hh)
#         modelu = utils.restore_model(modelu, f"obs_{hh}")
#         # utils.test_observer(modelu, f"obs_{hh}")
#         multi_obs[hh] = modelu
#         errs[(utils.K, hh)] = utils.compute_l2(modelu)
   
#     utils.mm_ode(multi_obs, lambdas)
#     utils.plot_weights(lambdas)

# utils.plot_l2_vs_k(errs)
# utils.plot_mm_observer(h_unk, gains, lambdas)
# errs_mm = utils.compute_mm_errors(h_unk, gains, lambdas)
# utils.plot_mm_l2_vs_k(errs_mm)

# errs_max = utils.compute_mm_max_errors(h_unk, gains, lambdas)
# utils.plot_mm_max_vs_k(errs_max)