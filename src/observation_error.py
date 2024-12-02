import os
import numpy as np
from omegaconf import OmegaConf
import hydra
# import common as co
import plots as pp
import coeff_calc as cc
from scipy import integrate
from utils import gen_testdata, calculate_mu, extract_matching
import time


np.random.seed(237)

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
conf_dir = os.path.join(src_dir, "configs")
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)

def prova_compute_mu(t, mu_par):
    """
    Compute mu with optional Gaussian noise.

    Parameters:
        t: Scalar or vector of time values.
        noise_std: Standard deviation of the Gaussian noise to be added. Default is 0 (no noise).

    Returns:
        muu: Array of computed mu values with added noise.
    """
    upsilon = cc.upsilon
    obs = cc.obs

    att=mu_par["attenuation"]
    noise_std=mu_par["noise"]
    offset=mu_par["offset"]

    # Compute mu for all t values simultaneously
    abs_diff = np.abs(obs[:, None] - cc.W_sys)  # Shape: (n_obs, 1)
    exp_t = np.exp(-att*t)  # If t is a scalar, exp_t is scalar; if t is a vector, shape is (len(t),)

    muu = t*upsilon * abs_diff * exp_t  # Broadcasting to compute for all t and all obs

    noise = np.random.normal(loc=0.0, scale=noise_std, size=muu.shape)
    muu += noise + offset

    return muu*(1-t)**5


def load_mu(matlab, tx, upsi):
    """
    Load mu values by finding the closest time(s) in matching_x0 to tx and computing differences.

    Parameters:
        matlab: Tuple containing the data to process.
        tx: Scalar or array of target times.

    Returns:
        mus[:, closest_indices]: Array of mu values corresponding to the closest times to tx.
    """
    # Extract matching data
    matching = extract_matching(matlab[0], *matlab[1])
    matching_x0 = matching[matching[:, 0] == 0][:, 1:]  # Time: matching_x0[:, 0], system: matching_x0[:, 1], observers

    # Time and observer data
    t = matching_x0[:, 0]
    mus = np.array([calculate_mu(matching_x0[:, 2 + i], matching_x0[:, 1], upsi) for i in range(cc.n_obs)])

    # Ensure tx is treated as an array
    tx = np.atleast_1d(tx)

    # Find the closest indices for each tx value
    closest_indices = np.abs(t[:, None] - tx).argmin(axis=0)
    res = mus[:, closest_indices]

    return res/np.max(res)

def solve_ivp(fold, matlab):
    """
    Solve the initial value problem (IVP) for observer weights.

    Parameters:
        fold (optional): Unused parameter (kept for compatibility).
        t_span (tuple): Time interval for the solution (default is (0, 1)).
        t_eval (array-like): Times at which to store the computed solution. Defaults to 100 points in t_span.
        p0 (array-like): Initial weights. Defaults to uniform weights.

    Returns:
        sol: OdeResult object containing the solution to the IVP.
    """
    n_obs = cc.n_obs
    lam = cc.lamb
    ups = cc.upsilon

    p0 = np.full((n_obs,), 1 / n_obs)  # Default: uniform weights

    t_eval = np.linspace(0, 1, 100)  # Default: 100 points in the time span

    def f(t, p):
        # a = compute_mu(t, mu_par)  # Shape (n_obs, len(t))
        a = load_mu(matlab, t, ups)
        e = np.exp(-a)     # Element-wise exponentiation

        weighted_sum = np.sum(p[:, None] * e, axis=0)  # Weighted sum for normalization

        # Vectorized computation for all elements
        f_matrix = -lam * (1 - (e / weighted_sum)) * p[:, None]
        return np.sum(f_matrix, axis=1)  # Summing along the second axis

    # Solve the IVP
    sol = integrate.solve_ivp(f, (0,1), p0, t_eval=t_eval)
    weights = np.zeros((sol.y.shape[0] + 1, sol.y.shape[1]))
    weights[0] = sol.t
    weights[1:] = sol.y

    # mu = compute_mu(weights[0], mu_par)
    mu = load_mu(matlab, weights[0], ups)
    
    np.savetxt(f"{fold}/weights_l_{lam:.3f}_u_{ups:.3f}.txt", weights.round(6), delimiter=' ')
    observers_mu = [
        {"t": weights[0], "weight": weights[i+1], "mu": mu[i], "label": f"observer_{i}"}
        for i in range(n_obs)
    ]

    string = f"l{lam}_u{ups}"
    # Plot results for multiple observers
    pp.plot_mu(observers_mu, fold, strng=string)
    pp.plot_weights(observers_mu, fold, strng=string)


conf = OmegaConf.load(f"{conf_dir}/config_run.yaml")
fold = f"{tests_dir}/simulation_mu_normalized"
os.makedirs(fold, exist_ok=True)

mu_params = {"attenuation": 5.0,
             "noise": 0.00005,
             "offset": 0.001
}

matlab = gen_testdata(conf,  path=f"{tests_dir}/cooling_simulation_8obs/ground_truth")
solve_ivp(fold, matlab)

# outputs = []
# params = [(1, 1), (100, 150)]

# for (lam, upsilon) in params:
#     start_time = time.time()
#     print("Starting simulation %d %d" % (lam, upsilon))
#     solve_ivp(lam, upsilon, fold, matlab)

#     exec_time = time.time() - start_time
#     print("--- %s seconds ---" % (round(exec_time, 3)))
#     outputs.append([lam, upsilon, round(exec_time, 3)])


# np.savetxt(f"{fold}/output.txt", np.array(outputs).round(3), delimiter=' ')