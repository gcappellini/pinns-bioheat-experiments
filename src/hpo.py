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
from omegaconf import OmegaConf

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

prj = "hpo_311024_obs0"
prj_figs = co.set_prj(prj)

# HPO setting
n_calls = 50
dim_learning_rate = Real(low=1e-5, high=5e-2, name="learning_rate", prior="log-uniform")
dim_num_dense_layers = Integer(low=3, high=10, name="num_dense_layers")
dim_num_dense_nodes = Integer(low=5, high=250, name="num_dense_nodes")
dim_activation = Categorical(categories=["elu", "silu", "sigmoid", "swish", "tanh"], name="activation") # ELU, GELU, ReLU, SELU, Sigmoid, SiLU, sin, Swish, tanh
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

default_parameters = [0.0001, 6, 92, "tanh", "Glorot normal"]

conf = OmegaConf.load(f"{src_dir}/config.yaml")

X, y_sys, y_obs, _ = uu.gen_testdata(conf, hpo=True)
x_obs = uu.gen_obsdata(conf, hpo=True)
tot_true = np.hstack((X, y_sys, y_obs))

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, num_dense_nodes, activation, initialization):
    global ITERATION
    run_figs = co.set_run(f"{ITERATION}")

    conf.model_parameters.n_obs = 1
    conf.model_properties.direct = False
    conf.model_properties.W = conf.model_parameters.W4
    conf.model_properties.activation, conf.model_properties.learning_rate, conf.model_properties.num_dense_layers = str(activation), learning_rate, int(num_dense_layers)
    conf.model_properties.num_dense_layers, conf.model_properties.initialization = int(num_dense_nodes), str(initialization)
    OmegaConf.save(conf, f"{run_figs}/config.yaml")

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


    # Generate and check observers if needed
    multi_obs = uu.mm_observer(conf)

    pred = multi_obs.predict(x_obs)
    data = np.column_stack((x_obs[:, 0].round(2), x_obs[:, -1].round(2), pred.round(4)))
    tot_pred = np.array(data)
    conf.output.save_figures=False
    # metrics = uu.check_observers_and_wandb_upload(tot_true, tot_pred, conf, run_figs)
    # metrics = uu.compute_metrics(tot_true[:, -1], tot_pred[:, -1], run_figs)

    error = np.sum(uu.calculate_l2(tot_pred[:, 0:2], tot_true[:, -1], tot_pred[:, -1]))
    metrics = {"l2_error": error}

    wandb.log(metrics)
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
    random_state=1444,
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