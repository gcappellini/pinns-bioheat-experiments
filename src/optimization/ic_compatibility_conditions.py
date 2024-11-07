import utils as uu
import os
import coeff_calc as cc
import plots as pp
import numpy as np
from omegaconf import OmegaConf
from scipy.optimize import minimize
import common as co

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)

def error_function(params, x_pinns_obs_ic, measurements_ic):
    b2, b3 = params  # The two parameters to optimize
    # Generate the predicted values using the function `uu.ic_obs`
    pinns_obs_ic = uu.ic_obs(x_pinns_obs_ic, b2=b2, b3=b3)
    
    # Compute the error (e.g., mean squared error)
    error = np.mean((pinns_obs_ic - measurements_ic) ** 2)
    
    return error

def find_best_params(x_pinns_obs_ic, measurements_ic, initial_guess=(10.0, 10.0)):
    # Use scipy.optimize.minimize to minimize the error
    result = minimize(error_function, initial_guess, args=(x_pinns_obs_ic, measurements_ic))
    
    # Extract the optimized parameters
    b2_opt, b3_opt = result.x
    return b2_opt, b3_opt

def new_ic(x):
    a = 0.1167
    b = -0.8167
    c = 0.7
    return a*x**2 + b*x + c

conf = OmegaConf.load(f"{src_dir}/config.yaml")
set = conf.experiment.name
rescale = conf.plot.rescale



# Find the best parameters starting from an initial guess
b2_opt, b3_opt = find_best_params(x_measurements_ic, measurements_ic, initial_guess=(1, 1))

print(f"exp {set} Optimized b2: {b2_opt.round(4)}, Optimized b3: {b3_opt.round(4)}")

def plot_t0(conf):
    rescale = conf.plot.rescale
    set = conf.experiment.name
    conf = uu.configure_settings(conf, set)
    OmegaConf.save(conf, f"{src_dir}/config.yaml")

    out_dir = co.set_prj(f"{set[0]}_{set[1]}/comparison")

    e = uu.import_testdata(f"{set[0]}_{set[1]}")
    measurements_ic = e[e[:, 1]==0][:,2]

    x_measurements_ic = uu.get_tc_positions()
    # conf_matlab = OmegaConf.load(f"{src_dir}/config_matlab.yaml")
    # uu.run_matlab_ground_truth(out_dir, conf_matlab, True)
    oo = np.hstack(uu.gen_testdata(conf))
    x_matlab_ic = np.unique(oo[:, 0])
    matlab_ic = oo[oo[:, 1]==0][:,-1]

    x_pinns = uu.gen_obsdata(conf)

    pinns_ic = uu.ic_obs(x_matlab_ic)

    x = [np.unique(x_pinns[:, 0]), x_matlab_ic, x_measurements_ic]
    y = [pinns_ic, matlab_ic, measurements_ic]
    title = "Comparison at t=0" if rescale else r"Comparison at $\tau=0$"
    xlabel, _, ylabel = uu.get_scaled_labels(rescale)
    legend_labels = ["Obs PINNs", "Obs MATLAB", "Measurements"]
    fname = f"{out_dir}/t0_{set[0]}_{set[1]}.png"
    sys_colors, mm_colors = uu.get_sys_mm_colors(conf)
    obs_colors = uu.get_obs_colors(conf)
    sys_linestyle, obs_linestyle = uu.get_sys_mm_linestyle(conf)
    colors = [obs_colors[1], mm_colors, sys_colors]
    linestyles = ["-", obs_linestyle, sys_linestyle]
    x_plot = uu.rescale_x(x) if rescale else x
    y_plot = uu.rescale_t(y) if rescale else y

    pp.plot_generic(x=x_plot,
                    y=y_plot,
                    title=title,
                    xlabel = xlabel,
                    ylabel=ylabel,
                    legend_labels=legend_labels,
                    filename=fname,
                    colors=colors,
                    linestyles=linestyles)

conf = OmegaConf.load(f"{src_dir}/config.yaml")
plot_t0(conf)