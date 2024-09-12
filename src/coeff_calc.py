import os
import numpy as np
from uncertainties import ufloat
import json
import hashlib

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)


with open(f"{src_dir}/properties.json", 'r') as f:
  data = json.load(f)

with open(f"{src_dir}/parameters.json", 'r') as f:
  data2 = json.load(f)

# Constants
L0, tauf = data["L0"], data["tauf"]
k, c, rho, h = data["k"], data["c"], data["rho"], data["h"]

Tmax, Troom, Twater, Ty20 = data["Tmax"], data["Troom"], data2["Twater"], data["Ty20"]   

W0, W1, W2, W3, W4, W5, W6, W7 = data2["W0"], data2["W1"], data2["W2"], data2["W3"], data2["W4"], data2["W5"], data2["W6"], data2["W7"]

K, lamb, delta = data["K"], data2["lambda"], data["delta"]

def scale_t(t):

    return (t - Troom) / (Tmax - Troom)

"coefficients a1, a2, a3, a4"

a1 = round((L0**2/tauf)*((rho*c)/k), 7)
a2 = round(L0**2*c/k, 7)
a3 = round(L0*h/k, 7)




