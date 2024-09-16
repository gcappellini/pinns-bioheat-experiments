import utils as uu
import common as co
from matplotlib import pyplot as plt
import numpy as np
import skopt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
import os
import deepxde as dde
import wandb
import plots as pp

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

dde.config.set_random_seed(200)

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

prj = "hpo_160924_obs5"
prj_figs = co.set_prj(prj)

# HPO setting
n_calls = 50
dim_learning_rate = Real(low=1e-4, high=5e-2, name="learning_rate", prior="log-uniform")
dim_num_dense_layers = Integer(low=1, high=6, name="num_dense_layers")
dim_num_dense_nodes = Integer(low=5, high=100, name="num_dense_nodes")
dim_activation = Categorical(categories=["elu", "relu", "selu", "silu", "sigmoid", "sin", "swish", "tanh"], name="activation")
dim_initialization = Categorical(categories=["Glorot normal", "Glorot uniform", "He normal", "He uniform"], name="initialization")
# dim_w_bc0 = Integer(low=1, high=1e+2, name="w_bc0", prior="log-uniform")

dimensions = [
    dim_learning_rate,
    dim_num_dense_layers,
    dim_num_dense_nodes,
    dim_activation,
    dim_initialization
    # dim_w_bc0
]

default_parameters = [0.001, 4, 50, "swish", "Glorot normal"]


@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, num_dense_nodes, activation, initialization):
    global ITERATION
    run_figs = co.set_run(f"{ITERATION}")

    config = uu.read_json(f"{src_dir}/properties.json")
    config["activation"] = activation
    config["learning_rate"] = learning_rate
    config["num_dense_layers"] = num_dense_layers
    config["num_dense_nodes"] = num_dense_nodes
    config["initialization"] = initialization

    aa = {"activation": activation,
          "learning_rate": learning_rate,
          "num_dense_layers": num_dense_layers,
          "num_dense_nodes": num_dense_nodes,
          "initialization": initialization
          }

    wandb.init(
        project=prj, name=f"{ITERATION}",
        config=aa
    )

    print(ITERATION, "it number")
    # Print the hyper-parameters.
    print("activation:", activation)
    print("initialization:", initialization)
    print("learning rate: {0:.1e}".format(learning_rate))
    print("num_dense_layers:", num_dense_layers)
    print("num_dense_nodes:", num_dense_nodes)
    # print("w_bc0:", w_bc0)
    print()

    data =  uu.read_json(f"{src_dir}/parameters.json")
    config["W"] = data["W5"]
    uu.write_json(config, f"{src_dir}/properties.json")
    uu.write_json(config, f"{run_figs}/properties.json")

    # Create the neural network with these hyper-parameters.
    model = uu.train_model(run_figs)
    xobs = uu.gen_obsdata()
    X, _, obs_true, _ =uu.gen_testdata()
    obs_pred = model.predict(xobs)
    pp.check_obs(X, obs_true[:, 5], obs_pred, 0, run_figs)

    errors = uu.compute_metrics(obs_true[:, 5], obs_pred)
    error = errors["L2RE"]

    wandb.log(errors)
    wandb.finish()

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
    random_state=1164,
)

print(search_result.x)

convergence_fig = plot_convergence(search_result)
convergence_fig.figure.savefig(f"{prj_figs}/convergence_plot.png")

# Plot objective and save the figure
plt.figure()
plot_objective(search_result, show_points=True, size=3.8)
plt.savefig(f"{prj_figs}/plot_obj.png",
            dpi=300, bbox_inches='tight')
plt.show()
plt.close()