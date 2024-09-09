import os
import numpy as np
from uncertainties import ufloat
import json

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)


with open(f"{src_dir}/properties.json", 'r') as f:
  data = json.load(f)

# Constants
L0, tauf = data["L0"], data["tauf"]
k, c, rho, h = data["k"], data["c"], data["rho"], data["h"]

Tmax, Troom = data["Tmax"], data["Troom"] 

W0, W1, W2, W3, W4, W5, W6, W7 = data["W0"], data["W1"], data["W2"], data["W3"], data["W4"], data["W5"], data["W6"], data["W7"]

K, lamb, delta = data["K"], data["lambda"], data["delta"]

"coefficients a1, a2, a3, a4"

a1 = round((L0**2/tauf)*((rho*c)/k), 7)
a2 = round(L0**2*c/k, 7)
a3 = round(L0*h/k, 7)

print(a1, a2, a3)




