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

dde.config.set_random_seed(200)

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")

prj = f"hpo_{datetime.date.today()}"

# HPO setting
n_calls = 50
dim_learning_rate = Real(high=0.19008887042142172, low=0.000004602124754177265, name="learning_rate", prior="log-uniform")
dim_num_dense_layers = Integer(low=2, high=26, name="num_dense_layers")
dim_num_dense_nodes = Integer(low=5, high=250, name="num_dense_nodes")
dim_activation = Categorical(categories=["ELU", "GELU", "ReLU", "SELU", "Sigmoid", "SiLU", "sin", "Swish", "tanh"], name="activation")
dim_initialization = Categorical(categories=["Glorot normal", "Glorot uniform", "He normal", "He uniform"], name="initialization")
# dim_w_bc0 = Integer(low=1, high=1e+2, name="w_bc0", prior="log-uniform")
# dim_w_bc1 = Integer(low=1, high=1e+2, name="w_bc1", prior="log-uniform")
# dim_w_res = Integer(low=1, high=1e+2, name="w_res", prior="log-uniform")
# dim_w_ic = Integer(low=1, high=1e+2, name="w_ic", prior="log-uniform")

dimensions = [
    dim_learning_rate,
    dim_num_dense_layers,
    dim_num_dense_nodes,
    dim_activation,
    dim_initialization,
    # dim_w_bc0,
    # dim_w_bc1,
    # dim_w_res,
    # dim_w_ic
]

default_parameters = [0.001, 4, 50, "tanh", "Glorot normal"]#, 1, 1, 1, 1]

conf_dir = os.path.join(src_dir, "configs")
conf = OmegaConf.load(f"{conf_dir}/config_run.yaml")
output_dir = conf.output_dir

system_gt, observers_gt, mm_obs_gt = uu.gen_testdata(conf, path=f"{tests_dir}/cooling_simulation")
x_obs = uu.gen_obsdata(conf, path=f"{tests_dir}/cooling_simulation")

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, num_dense_nodes, activation, initialization):#, w_bc0, w_bc1, w_ic, w_res):
    global ITERATION

    conf.model_properties.activation, conf.model_properties.learning_rate, conf.model_properties.num_dense_layers = str(activation), learning_rate, int(num_dense_layers)
    conf.model_properties.num_dense_nodes, conf.model_properties.initialization = int(num_dense_nodes), str(initialization)
    #conf.model_properties.w_bc0, conf.model_properties.w_bc1, conf.model_properties.w_ic, conf.model_properties.w_res = int(w_bc0), int(w_bc1), int(w_ic), int(w_res) 
    run_figs = co.set_run(output_dir, conf, f"hpo_{ITERATION}")

    aa = {"activation": activation,
          "learning_rate": learning_rate,
          "num_dense_layers": num_dense_layers,
          "num_dense_nodes": num_dense_nodes,
          "initialization": initialization,
        #   "w_bc0": w_bc0,
        #   "w_bc1": w_bc1,
        #   "w_ic": w_ic,
        #   "w_res": w_res
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
    # print("w_bc1:", w_bc1)
    # print("w_ic:", w_ic)
    # print("w_res:", w_res)

    print()


    # Generate and check observers if needed
    multi_obs = uu.mm_observer(conf)

    pred = multi_obs.predict(x_obs)

    error = np.sum(uu.calculate_l2(system_gt["grid"], mm_obs_gt["theta"], pred))
    metrics_tot = uu.compute_metrics(system_gt["grid"], mm_obs_gt["theta"], pred, run_figs, system=system_gt["theta"])
    metrics = {"L2RE": error,
               "L2RE_sys": metrics_tot["total_L2RE_sys"]}

    wandb.log(metrics)
    wandb.finish()

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

partial_dependence


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