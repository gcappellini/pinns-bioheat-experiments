


import utils as uu
import common as co
from matplotlib import pyplot as plt
import numpy as np
import skopt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective, plot_regret, partial_dependence, plot_evaluations, plot_histogram
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
import os
import deepxde as dde
import wandb
import plots as pp
from omegaconf import OmegaConf
import datetime

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
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")

prj = f"hpo_{datetime.date.today()}"

# HPO setting
n_calls = 50
dim_learning_rate = Real(high=0.01, low=0.00005, name="learning_rate", prior="log-uniform")
dim_num_dense_layers = Integer(low=2, high=8, name="num_dense_layers")
dim_num_dense_nodes = Integer(low=5, high=250, name="num_dense_nodes")
dim_activation = Categorical(categories=["ELU", "GELU", "ReLU", "SELU", "Sigmoid", "SiLU", "sin", "Swish", "tanh"], name="activation")
dim_initialization = Categorical(categories=["Glorot normal", "Glorot uniform", "He normal", "He uniform"], name="initialization")

dimensions = [
    dim_learning_rate,
    dim_num_dense_layers,
    dim_num_dense_nodes,
    dim_activation,
    dim_initialization
]

default_parameters = [0.001, 4, 50, "tanh", "Glorot normal"]

conf_dir = os.path.join(src_dir, "configs")
conf = OmegaConf.load(f"{conf_dir}/config_run.yaml")
output_dir = conf.output_dir
props = conf.model_properties

system_gt, observers_gt, mm_obs_gt = uu.gen_testdata(conf, path=f"{tests_dir}/cooling_simulation")
x_obs = uu.gen_obsdata(conf, path=f"{tests_dir}/cooling_simulation")

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, num_dense_nodes, activation, initialization):
    global ITERATION

    props.activation = str(activation)
    props.learning_rate = learning_rate
    props.num_dense_layers = int(num_dense_layers)
    props.num_dense_nodes = int(num_dense_nodes)
    props.initialization = str(initialization)

    run_figs = co.set_run(output_dir, conf, f"hpo_{ITERATION}")

    config = {
        "activation": activation,
        "learning_rate": learning_rate,
        "num_dense_layers": num_dense_layers,
        "num_dense_nodes": num_dense_nodes,
        "initialization": initialization
    }

    with wandb.init(project=prj, name=f"{ITERATION}", config=config):
        print(f"Iteration {ITERATION}")
        print(f"activation: {activation}")
        print(f"initialization: {initialization}")
        print(f"learning rate: {learning_rate:.1e}")
        print(f"num_dense_layers: {num_dense_layers}")
        print(f"num_dense_nodes: {num_dense_nodes}")

        # Generate and check observers if needed
        multi_obs = uu.mm_observer(conf)
        pred = multi_obs.predict(x_obs)

        error = np.sum(uu.calculate_l2(system_gt["grid"], mm_obs_gt["theta"], pred))
        metrics_tot = uu.compute_metrics(system_gt["grid"], mm_obs_gt["theta"], pred, run_figs, system=system_gt["theta"])
        metrics = {"L2RE": error, "L2RE_sys": metrics_tot["total_L2RE_sys"]}

        wandb.log(metrics)

    if np.isnan(error):
        error = 10**5

    ITERATION += 1
    return error

ITERATION = 0

res = gp_minimize(
    func=fitness,
    dimensions=dimensions,
    acq_func="EI",  # Expected Improvement.
    n_calls=n_calls,
    x0=default_parameters,
    random_state=1444,
)

print(res.x)

convergence_fig = plot_convergence(res)
convergence_fig.figure.savefig(f"{output_dir}/convergence_plot.png")

# Plot objective and save the figure
plt.figure()
plot_objective(res, show_points=True, size=3.8)
plt.savefig(f"{output_dir}/plot_obj.png",
            dpi=300, bbox_inches='tight')
plt.show()
plt.close()

partial_dependence_fig = partial_dependence(res)
partial_dependence_fig.figure.savefig(f"{output_dir}/partial_dependenc.png")

convergence_fig = plot_convergence(res)
convergence_fig.figure.savefig(f"{output_dir}/convergence_plot.png")

convergence_fig = plot_objective(res)
convergence_fig.figure.savefig(f"{output_dir}/objective_function.png")

convergence_fig = plot_evaluations(res)
convergence_fig.figure.savefig(f"{output_dir}/evaluations.png")

file_path = os.path.join(output_dir, 'results_hpo.txt')

with open(file_path, 'w') as file:
    file.write("Named Tuple Results:\n")
    for field, value in res.items():
        file.write(f"{field}: {value}\n")
