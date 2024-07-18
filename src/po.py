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

n="measurements/vessel/0"
prj = "optim_properties_obs"
prj_figs, _, _ = utils.set_prj(prj)
# HPO setting
n_calls = 50
dim_a1 = Real(low=1e-2, high=5e+2, name="a1", prior="log-uniform")
dim_a2 = Real(low=1e-2, high=5e+2, name="a2", prior="log-uniform")
dim_a3 = Real(low=1e-2, high=5e+2, name="a3", prior="log-uniform")
dim_a4 = Real(low=1e-2, high=5e+2, name="a4", prior="log-uniform")
dim_a5 = Real(low=1e-2, high=5e+2, name="a5", prior="log-uniform")
dim_a6 = Real(low=1e-2, high=5e+2, name="a6", prior="log-uniform")

dimensions = [
    dim_a1,
    dim_a2,
    dim_a3,
    dim_a4,
    dim_a5,
    dim_a6
]

default_parameters = [cc.a1, cc.a2, cc.a3, cc.a4, cc.a5, cc.a6]


@use_named_args(dimensions=dimensions)
def fitness(a1, a2, a3, a4, a5, a6):
    global ITERATION
    run = f"run_{ITERATION}"
    utils.set_run(run)

    config = utils.read_json("properties.json")
    config["a1"] = a1
    config["a2"] = a2
    config["a3"] = a3
    config["a4"] = a4
    config["a5"] = a5
    config["a6"] = a6
    utils.write_json(config)

    print(ITERATION, "it number")
    # Print the hyper-parameters.
    print("a1:", a1)
    print("a2:", a2)
    print("a3:", a3)
    print("a4:", a4)
    print("a5:", a5)
    print("a6:", a6)
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