from utils import scale_t, ic_obs, run_matlab_ground_truth
import os
import numpy as np
from omegaconf import OmegaConf
from coeff_calc import Ty10, Ty20, Ty30, a5
np.random.seed(237)
import matplotlib.pyplot as plt
from skopt import gp_minimize
from scipy.optimize import minimize
from skopt.plots import plot_gaussian_process, plot_convergence, plot_objective, plot_evaluations
from skopt.space import Real
from skopt.utils import use_named_args
from common import set_run
import plots as pp

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
conf_dir = os.path.join(src_dir, "configs")
os.makedirs(tests_dir, exist_ok=True)

conf = OmegaConf.load(f"{conf_dir}/config_run.yaml")
output_dir = conf.output_dir
x = np.linspace(0, 1, num=100)

K = conf.model_properties.K

b1_0= 6.3
b2_0= 0.9
y1_0, y2_0, y3_0 = scale_t(Ty10), scale_t(Ty20), scale_t(Ty30)

dim_b1 = Real(low=0.0, high=10.0, name="b1")
dim_b2 = Real(low=0.0, high=1.0, name="b2")

dimensions = [
    dim_b1, 
    dim_b2
]

iters_history = {}

@use_named_args(dimensions=dimensions)
def fitness(b1, b2):
    global ITERATION
    # ITERATION: str="BEST"
    run_figs = set_run(output_dir, f"{ITERATION}")
            
    conf.model_properties.b1 = float(b1)
    conf.model_properties.b2 = float(b2)
    OmegaConf.save(conf, f"{run_figs}/config.yaml")
    OmegaConf.save(conf, f"{src_dir}/config.yaml")

    print(ITERATION, "it number")
    print("b1:", b1)
    print("b2:", b2)

    print()

    metr = run_matlab_ground_truth(run_figs)
    iters_history[ITERATION] = {"b1": b1, "b2": b2, "fitness": metr}

    ITERATION += 1

    return metr


# res = gp_minimize(fitness,                  # the function to minimize
#                   dimensions=dimensions,      # the bounds on each dimension of x
#                   acq_func="gp_hedge",      # the acquisition function
#                   n_calls=50,         # the number of evaluations of f
#                   n_random_starts=5,  # the number of random initialization points
#                   x0 = [b1_0, b2_0],
#                 #   noise=0.1**2,       # the noise level (optional)
                
#                   random_state=1234) 


# convergence_fig = plot_convergence(res)
# convergence_fig.figure.savefig(f"{output_dir}/convergence_plot.png")

# convergence_fig = plot_objective(res)
# convergence_fig.figure.savefig(f"{output_dir}/objective_function.png")

# convergence_fig = plot_evaluations(res)
# convergence_fig.figure.savefig(f"{output_dir}/evaluations.png")

# file_path = os.path.join(output_dir, 'results_ic.txt')

# with open(file_path, 'w') as file:
#     file.write("Named Tuple Results:\n")
#     for field, value in res.items():
#         file.write(f"{field}: {value}\n")

# print(fitness(4.9149, 0.3540))
ITERATION = 0

bounds = [(0, 10), (0, 1)]
result = minimize(fitness, [b1_0, b2_0], bounds=bounds, method='L-BFGS-B')

print("x^*=(%.4f, %.4f) f(x^*)=%.4f" % (result.x[0], result.x[1], result.fun))

file_path = os.path.join(output_dir, 'results_ic.txt')
with open(file_path, 'w') as file:
    file.write("Optimization Results:\n")
    file.write(f"Optimal values of x and y: {result.x}\n")
    file.write(f"Minimum value of the function: {result.fun}\n")
    file.write(f"Gradients at optimal point (computed automatically): {result.jac}\n")
    file.write(f"Number of function evaluations: {result.nfev}\n")
    file.write(f"Number of gradient evaluations: {result.njev}\n")

        # Write the iteration history
    file.write("\nIteration History:\n")
    file.write("Iteration, b1, b2, Fitness\n")
    for iteration, data in iters_history.items():
        file.write(f"{iteration}, {data['b1']}, {data['b2']}, {data['fitness']}\n")


print("Optimal values of x and y:", result.x)
print("Minimum value of the function:", result.fun)
print("Gradients at optimal point (computed automatically):", result.jac)
print("Number of function evaluations:", result.nfev)
print("Number of gradient evaluations:", result.njev)

iterations = list(iters_history.keys())  # Get iteration numbers (or use a sequence of values)
fitness_values = [data["fitness"] for data in iters_history.values()]

pp.plot_generic(iterations, fitness_values, 'Convergence Plot', 'Iteration', 'Fitness value', f"{output_dir}/convergence_plot.png")


