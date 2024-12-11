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

y1_0, y2_0, y3_0 = scale_t(props.Ty10), scale_t(props.Ty20), scale_t(props.Ty30)
a5 = 1.1666667
alfa, L0 = props.alfa, props.L0
K = alfa*L0

iters_history = {}

for el in range(len(possible_b1)):
    b1 = float(possible_b1[el])
    props.b1 = b1
    a = np.array([[1+K*b1, 1],[b1-1, (b1-1)*np.exp(K)]])
    b = np.array([(a5*y3_0+(K-a5)*y2_0), y1_0])
    resu = np.linalg.solve(a,b).round(5)
    [b2, b3] = resu
    hat_theta_0 = b1*(b2+b3)
    print(f"Iteration {el}: b1={b1}, thetahat_0={hat_theta_0}")
    props.b2, props.b3 = float(b2), float(b3)
    out_dir = f"{fold}/{el}"
    os.makedirs(out_dir, exist_ok=True)
    output_dir_gt, config_matlab = set_run(out_dir, conf, "ground_truth")
    run_matlab_ground_truth()
    system_gt, observers_gt, mm_obs_gt = gen_testdata(conf, path=output_dir_gt)
    metric = calculate_l2(system_gt["grid"], system_gt["theta"], mm_obs_gt["theta"])
    iters_history[el] = {"b1": b1, "theta_hat_0": hat_theta_0, "fitness": norm(metric)}

sorted_iters_history = dict(
    sorted(iters_history.items(), key=lambda item: item[1]["fitness"], reverse=True)
)

file_path = os.path.join(out_dir, 'results_ic.txt')


with open(file_path, 'w') as file:    
    for iteration, data in sorted_iters_history.items():
        file.write(f"{iteration}, {data['b1']}, {data['theta_hat_0']}, {data['fitness']}\n")


iterations = list(iters_history.keys())  # Get iteration numbers (or use a sequence of values)
fitness_values = [data["fitness"] for data in iters_history.values()]


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
theta_hat_0_values = [data["theta_hat_0"] for data in iters_history.values()]

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
plt.plot(iterations, theta_hat_0_values, marker='o', color='g', label='b2')
plt.xlabel('Iteration')
plt.ylabel(r'$\hat \theta_0 (0)$')
plt.title(r'$\hat \theta_0 (0)$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{fold}/theta_hat_0.png")
# plt.show()
