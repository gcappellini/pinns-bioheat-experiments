from utils import scale_t, ic_obs, run_matlab_ground_truth
import os
import numpy as np
from omegaconf import OmegaConf
from coeff_calc import Ty10, Ty20, Ty30, a5
np.random.seed(237)
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.plots import plot_gaussian_process
from skopt.space import Real
from skopt.utils import use_named_args
from common import set_run, set_prj

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)

conf = OmegaConf.load(f"{src_dir}/config.yaml")
x = np.linspace(0, 1, num=100)

K = conf.model_properties.K

# def ic_obs(x, b1, b2):
#     return (1-x**b1)*(np.exp(-50/(x+0.001))+b2)

# def der_ic_obs(x, b1, b2):
#     val = -b1*(x**(b1 - 1)) * (np.exp(-(50)/(x + 0.001)) + b2) + (50 *(1 - x**(b1)) * np.exp(-(50)/(x + 0.001)))/(x + 0.001)**2
#     return val

# def new_ic_obs(x, obs0):
#     A = a5*(y3_0 - y2_0)
#     B = y2_0
#     C = obs0 - (A - K*B)/K
#     return ((A-K*B)/K+C*np.exp(K*x))*(1-x)**(2.8)


# x^*=(1.4, 0.818) f(x^*)=1.3095 ottima convergenza
# x^*=(0.90, 0.818) f(x^*)=0.1445 
# x^*=(0.8912, 0.8500) f(x^*)=0.0100
# x^*=(0.8927, 0.8500) f(x^*)=0.0001 with gp_edge
# x^*=(0.8771, 0.8180) f(x^*)=0.0007 with gp_edge
# x^*=(0.7956, 0.6436) f(x^*)=0.0547 Troppo distante, osservatore non converge
# x^*=(0.8416, 0.7514) f(x^*)=0.0161
# x^*=(0.8072, 0.8146) f(x^*)=0.1181 per K=5
b1_0= 6.3
b2_0= 0.9
y1_0, y2_0, y3_0 = scale_t(Ty10), scale_t(Ty20), scale_t(Ty30)

dim_b1 = Real(low=0.0, high=10.0, name="b1")
dim_b2 = Real(low=0.0, high=1.0, name="b2")

dimensions = [
    dim_b1, 
    dim_b2
]

# @use_named_args(dimensions=dimensions)
def fitness(b1, b2):
    global ITERATION
    # ITERATION: str="BEST"
    run_figs = set_run(f"{ITERATION}")
            
    conf.model_properties.b1 = b1
    conf.model_properties.b2 = b2
    OmegaConf.save(conf, f"{run_figs}/config.yaml")
    OmegaConf.save(conf, f"{src_dir}/config.yaml")

    print(ITERATION, "it number")
    print("b1:", b1)
    print("b2:", b2)

    print()

    metr = run_matlab_ground_truth(run_figs)

    ITERATION += 1

    return metr

ITERATION = 0
out_dir = set_prj(conf.output_dir)

# res = gp_minimize(fitness,                  # the function to minimize
#                   dimensions=dimensions,      # the bounds on each dimension of x
#                   acq_func="gp_hedge",      # the acquisition function
#                   n_calls=50,         # the number of evaluations of f
#                   n_random_starts=5,  # the number of random initialization points
#                   x0 = [b1_0, b2_0],
#                 #   noise=0.1**2,       # the noise level (optional)
                
#                   random_state=1234) 


# print("x^*=(%.4f, %.4f) f(x^*)=%.4f" % (res.x[0], res.x[1], res.fun))

print(fitness(4.9149, 0.3540))

