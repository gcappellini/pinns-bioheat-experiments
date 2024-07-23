import utils_meas as utils
from matplotlib import pyplot as plt
import numpy as np
import skopt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
import os
import deepxde as dde
import coeff_calc as cc


# Function 'gp_minimize' of package 'skopt(scikit-optimize)' is used in this example.
# However 'np.int' used in skopt 0.9.0(the latest version) was deprecated since NumPy 1.20.
# Monkey patch here to fix the error.
np.int = int

if dde.backend.backend_name == "pytorch":
    sin = dde.backend.pytorch.sin
elif dde.backend.backend_name == "paddle":
    sin = dde.backend.paddle.sin
else:
    from deepxde.backend import tf

    sin = tf.sin



dde.config.set_random_seed(300)
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
project_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(project_dir, "tests")
figures = os.path.join(tests_dir, "figures")
os.makedirs(figures, exist_ok=True)

n="3"
prj = "2nd_try_po_obs"
prj_figs, _, _ = utils.set_prj(prj)
# HPO setting
n_calls = 50
# dim_a1 = Real(low=5e-1, high=1e+1, name="a1", prior="log-uniform")
dim_a2 = Real(low=5e+1, high=2e+2, name="a2", prior="log-uniform")
# dim_a3 = Real(low=5e-1, high=1e+1, name="a3", prior="log-uniform")
# dim_a5 = Real(low=1e+1, high=5e+2, name="a5", prior="log-uniform")
# dim_a6 = Real(low=1e-1, high=1e+2, name="a6", prior="log-uniform")

dimensions = [
    # dim_a1,
    dim_a2,
    # dim_a3,
    # dim_a5,
    # dim_a6
]

# default_parameters = [cc.a1, cc.a2, cc.a3, cc.a5, cc.a6]
default_parameters = [cc.a2]


@use_named_args(dimensions=dimensions)
# def fitness(a1, a2, a3, a5, a6):
def fitness(a2):
    global ITERATION
    run = f"run_{ITERATION}"
    utils.set_run(run)

    config = utils.read_json("properties.json")
    # config["a1"] = round(a1, 7)
    config["a2"] = round(a2, 7)
    # config["a3"] = round(a3, 7)
    # config["a5"] = round(a5, 7)
    # config["a6"] = round(a6, 7)
    utils.write_json(config, "properties.json")

    print(ITERATION, "it number")
    # Print the hyper-parameters.
    # print("a1:", round(a1, 7))
    print("a2:", round(a2, 7))
    # print("a3:", round(a3, 7))
    # print("a5:", round(a5, 7))
    # print("a6:", round(a6, 7))
    print()

    # Create the neural network with these hyper-parameters.
    mo, errors = utils.single_observer(prj, run, n)

    error = errors["L2RE"]

    if np.isnan(error):
        error = 10**5

    ITERATION += 1
    return error


ITERATION = 0

search_result = gp_minimize(
    func=fitness,
    dimensions=dimensions,
    acq_func="EI",  # Expected Improvement.
    n_calls=n_calls,
    x0=default_parameters,
    random_state=1234,
)

print(search_result.x)

convergence_fig = plot_convergence(search_result)
convergence_fig.figure.savefig(f"{figures}/convergence_plot_{prj}.png")

# Plot objective and save the figure
plt.figure()
plot_objective(search_result, show_points=True, size=3.8)
plt.savefig(f"{prj_figs}/plot_obj.png",
            dpi=300, bbox_inches='tight')
plt.show()