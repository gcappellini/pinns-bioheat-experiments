import matlab.engine
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
k, cfl, rho, h = data["k"], data["c"], data["rho"], data["h"]

T_tumour, Ttis, Tw = data["Tmax"], data["Ttis"], data["Tw"] # tumour temperature in °C
T_fluid_initial = Ttis
T_fluid_end = Ttis + .80

P0, d, a, b, x0, beta = data["P0"], data["d"], data["a"], data["b"], data["x0"], data["beta"]
P0_x = beta* P0 * np.exp(a*x0)


R1 = 0.001

tfl = T_fluid_initial
tr2 = 24.0
Nu, K = 4.01, 1.0

out = tfl+(tr2-tfl)*(1+0.5*Nu*K*np.log(0.01/R1))

print(out)


