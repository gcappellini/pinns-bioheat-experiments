import numpy as np
import utils

U = 369.0


n_observers = 8
# gains =  [1, 4, 20, 100, 500, 10000]
gain = 100

lambdas = [10, 200, 1000]

U_unk = np.linspace(U/2, U, num=n_observers).round(1)

utils.set_K(gain)

multi_obs, errs = utils.create_mm_observer(U_unk, gain)

    # utils.plot_l2_vs_k(errs)
# utils.plot_l2_vs_u(errs)

utils.mm_ode(multi_obs, lambdas)
utils.plot_weights(lambdas)


# utils.plot_continuous(100, 369)


utils.plot_mm_observer(U_unk, [gain], lambdas)

# utils.plot_mm_observer(h_unk, gains, lambdas)
# errs_mm = utils.compute_mm_errors(h_unk, gains, lambdas)
# utils.plot_mm_l2_vs_k(errs_mm)

# errs_max = utils.compute_mm_max_errors(h_unk, gains, lambdas)
# utils.plot_mm_max_vs_k(errs_max)