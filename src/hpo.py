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
from hydra import compose
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
conf = compose(config_name="config_run")
output_dir = conf.output_dir
props = conf.model_properties
pars = conf.model_parameters
gt_path=f"{tests_dir}/cooling_ground_truth_5e-04"

prj = f"{datetime.date.today()}_Obs{pars.W_index}"

system_gt, observers_gt, mm_obs_gt = uu.gen_testdata(conf, path=gt_path)
x_obs = uu.gen_obsdata(conf, system_gt)

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, num_dense_nodes, activation, initialization):
    global ITERATION

    props.activation = str(activation)
    props.learning_rate = learning_rate
    props.num_dense_layers = int(num_dense_layers)
    props.num_dense_nodes = int(num_dense_nodes)
    props.initialization = str(initialization)

    run_figs, _ = co.set_run(output_dir, conf, f"hpo_{ITERATION}")

    config_wb = {
        "activation": activation,
        "learning_rate": learning_rate,
        "num_dense_layers": num_dense_layers,
        "num_dense_nodes": num_dense_nodes,
        "initialization": initialization
    }
    label = f"{ITERATION}"

    with wandb.init(project=prj, name=label, config=config_wb):
        print(f"Iteration {ITERATION}")
        print(f"activation: {activation}")
        print(f"initialization: {initialization}")
        print(f"learning rate: {learning_rate:.1e}")
        print(f"num_dense_layers: {num_dense_layers}")
        print(f"num_dense_nodes: {num_dense_nodes}")

        # Generate and check observers if needed
        multi_obs, error = uu.execute(conf, label)
        _, obs_pred = uu.get_observers_preds(system_gt, multi_obs, x_obs, run_figs, conf, "simulation")

        _, obs_pred = uu.calculate_l2(mm_obs_gt, [], obs_pred)
        # error = np.sum(obs_pred["L2_err"])
        metrics_tot = uu.compute_metrics([mm_obs_gt, obs_pred], conf, run_figs)
        metrics_tot = {key.replace(f"observer_{pars.W_index}_", ""): value for key, value in metrics_tot.items()}
        # error = metrics_tot["L2RE"]
        metrics_tot["test"] = error
   
        wandb.log(metrics_tot)
    pp.plot_multiple_series([obs_pred, mm_obs_gt], run_figs, label)
    pp.plot_l2(system_gt, [obs_pred, mm_obs_gt], run_figs, label)
    pp.plot_obs_err([obs_pred, mm_obs_gt], run_figs, label)


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

# partial_dependence_fig = partial_dependence(res)
# partial_dependence_fig.figure.savefig(f"{output_dir}/partial_dependenc.png")

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
