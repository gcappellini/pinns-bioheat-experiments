import utils
import os
import json
import numpy as np
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

with open(f"{src_dir}/properties.json", 'r') as f:
  data = json.load(f)

# Constants
L, L0 = data["L"], data["L0"]
k, cfl, rho, h = data["k"], data["c"], data["rho"], data["h"]

Tmax, Troom, Tw, dT= data["Tmax"], data["Troom"], data["Tw"], data["dT"] # tumour temperature in °C
SAR0, d, a, b, x0, beta = data["SAR0"], data["d"], data["a"], data["b"], data["x0"], data["beta"]

W1, W2, W3, alpha, eta = data["W1"], data["W2"], data["W3"], data["alpha"], data["eta"]


prj = "prova"
n = 2
run = "prova"


utils.mm_observer(prj, n)