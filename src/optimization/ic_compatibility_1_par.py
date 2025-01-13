from utils import scale_t, run_matlab_ground_truth, gen_testdata, calculate_l2
import os
import numpy as np
from numpy.linalg import norm
from omegaconf import OmegaConf
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
fold = f"{tests_dir}/optimization_ic"
os.makedirs(fold, exist_ok=True)
possible_b1 = np.linspace(0, 10, num=100).round(3)
possible_b1 = possible_b1[possible_b1 != 1]

props = conf.model_properties

iters_history = {}



# Helper function to compute the fitness for a given b1
def compute_fitness(b1, props, conf, fold):
    global ITERATION
    y1_0, y2_0, y3_0 = scale_t(props.Ty10), scale_t(props.Ty20), scale_t(props.Ty30)
    a5 = 1.1666667
    alfa, L0 = props.alfa, props.L0
    K = alfa*L0
    # Update properties
    props.b1 = b1
    a = np.array([[1 + K * b1, 1], [b1 - 1, (b1 - 1) * np.exp(K)]])
    b = np.array([(a5 * y3_0 + (K - a5) * y2_0), y1_0])
    
    try:
        resu = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.inf  # Return a large fitness value if the system is unsolvable
    
    [b2, b3] = resu
    props.b2, props.b3 = float(b2), float(b3)
    hat_theta_0 = b1 * (b2 + b3)
    print(f"Iteration {ITERATION}: b1={b1}, thetahat_0={hat_theta_0}")
    
    # Generate test data and compute L2 norm
    output_dir_gt, config_matlab = set_run(fold, conf, "ground_truth")
    run_matlab_ground_truth()
    system_gt, observers_gt, mm_obs_gt = gen_testdata(conf, path=output_dir_gt)
    metric = calculate_l2(system_gt["grid"], system_gt["theta"], mm_obs_gt["theta"])
    fitness = norm(metric)
    iters_history[ITERATION] = {"b1": b1, "thetahat_0": hat_theta_0, "metric": fitness}
    ITERATION += 1
    
    return fitness

# Define the objective function for optimization
def objective_function(b1, props, conf, fold):
    # Ensure b1 is a scalar
    b1 = float(b1)
    if np.isclose(b1, 1.0):
        return np.inf  # Penalize invalid b1 values
    return compute_fitness(b1, props, conf, fold)

# Optimization setup
ITERATION = 0
bounds = [(0, 10)]  # Bound b1 between 0 and 10
initial_guess = [5.0]  # Start the search at b1 = 5.0

# Run the optimizer
result = minimize(
    objective_function,
    x0=initial_guess,
    args=(props, conf, fold),
    method="L-BFGS-B",
    bounds=bounds
)

# Output results
optimal_b1 = result.x[0]
optimal_fitness = result.fun

print(f"Optimal b1: {optimal_b1}, Fitness: {optimal_fitness}")

iterations = list(iters_history.keys())  # Get iteration numbers (or use a sequence of values)
fitness_values = [data["metric"] for data in iters_history.values()]

# pp.plot_generic(iterations, fitness_values, 'Convergence Plot', 'Iteration', 'Fitness value', f"{output_dir}/convergence_plot.png")

# Plot the convergence
plt.figure(figsize=(8, 6))
plt.plot(iterations, fitness_values, marker='o', color='b', label='Fitness value')
plt.xlabel('Iteration')
plt.ylabel('L2 norm')
plt.title('Convergence Plot')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the convergence plot
plt.savefig(f"{fold}/convergence_plot.png")


# Extract b1 and b2 values for each iteration
b1_values = [data["b1"] for data in iters_history.values()]
b2_values = [data["thetahat_0"] for data in iters_history.values()]

# Plot the convergence of b1
plt.figure(figsize=(8, 6))
plt.plot(iterations, b1_values, marker='o', color='r', label='b1')
plt.xlabel('Iteration')
plt.ylabel('b1 Value')
plt.title('Convergence of b1')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{fold}/b1_convergence_plot.png")
# plt.show()

# Plot the convergence of b2
plt.figure(figsize=(8, 6))
plt.plot(iterations, b2_values, marker='o', color='g', label='b2')
plt.xlabel('Iteration')
plt.ylabel(r'$\hat \theta_0$ Value')
plt.title(r'Convergence of $\hat \theta_0$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{fold}/thetahat_0_convergence_plot.png")


file_path = os.path.join(fold, 'results_ic.txt')
with open(file_path, 'w') as file:
    file.write("Optimization Results:\n")
    file.write(f"Optimal values of x and y: {result.x}\n")
    file.write(f"Minimum value of the function: {result.fun}\n")
    file.write(f"Gradients at optimal point (computed automatically): {result.jac}\n")
    file.write(f"Number of function evaluations: {result.nfev}\n")
    file.write(f"Number of gradient evaluations: {result.njev}\n")