
import os
import matplotlib.pyplot as plt
import numpy as np
import json

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)

# eng = matlab.engine.start_matlab()
# eng.cd(src_dir, nargout=0)
# eng.simple_script(nargout=0)
# eng.quit()

with open(f"{src_dir}/properties.json", 'r') as f:
  data = json.load(f)

# Constants
L, L0 = data["L"], data["L0"]
k, cfl, rho, h, W1, W2 = data["k"], data["c"], data["rho"], data["h"], data["W1"], data["W2"]

T_tumour, Ttis, Tw = data["Tmax"], data["Ttis"], data["Tw"] # tumour temperature in °C
T_fluid_initial = Ttis
T_fluid_end = Ttis + .80

P0, d, a, b, x0, beta = data["P0"], data["d"], data["a"], data["b"], data["x0"], data["beta"]
P0_x = beta* P0 * np.exp(a*x0)

r = 0.5 / 1000  # radius in meters
v = 1.0 / 100  # velocity in m/s

R1_vessel2 = 1.0 / 1000  # radius in meters
v_vessel2 = 1.5 / 100  # velocity in m/s


e = np.load(f"{src_dir}/data/simulations/simulation/y_axis_{r}_{v}_{W1}.npz", allow_pickle=True)

print(e["vessel"][50])