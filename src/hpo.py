import utils_cdc as utils
from matplotlib import pyplot as plt
import numpy as np
import skopt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
import os
import deepxde as dde

dde.config.set_random_seed(100)

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

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
project_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(project_dir, "tests")
figures = os.path.join(tests_dir, "figures")
os.makedirs(figures, exist_ok=True)

n="0-cdc"
prj = "single_obs_hpo"
prj_figs, _, _ = utils.set_prj(prj)
# HPO setting
n_calls = 50
dim_learning_rate = Real(low=1e-4, high=5e-2, name="learning_rate", prior="log-uniform")
dim_num_dense_layers = Integer(low=1, high=6, name="num_dense_layers")
dim_num_dense_nodes = Integer(low=5, high=100, name="num_dense_nodes")
dim_activation = Categorical(categories=["elu", "relu", "selu", "silu", "sigmoid", "sin", "swish", "tanh"], name="activation")

dimensions = [
    dim_learning_rate,
    dim_num_dense_layers,
    dim_num_dense_nodes,
    dim_activation,
]

default_parameters = [1e-3, 4, 50, "tanh"]


@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, num_dense_nodes, activation):
    global ITERATION
    run = f"run_{ITERATION}"
    utils.set_run(run)

    config = utils.read_config()
    config["activation"] = activation
    config["learning_rate"] = learning_rate
    config["num_dense_layers"] = num_dense_layers
    config["num_dense_nodes"] = num_dense_nodes
    utils.write_config(config)

    print(ITERATION, "it number")
    # Print the hyper-parameters.
    print("activation:", activation)
    print("learning rate: {0:.1e}".format(learning_rate))
    print("num_dense_layers:", num_dense_layers)
    print("num_dense_nodes:", num_dense_nodes)
    print()

    # Create the neural network with these hyper-parameters.
    mo, _ = utils.single_observer(prj, run, n)
    error = utils.metrics_observer(mo, n)

    if np.isnan(error):
        error = 10**5

    ITERATION += 1
    return error


ITERATION = 0

search_result = gp_minimize(
    func=fitness,
    dimensions=dimensions,
    # acq_func="EI",  # Expected Improvement.
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