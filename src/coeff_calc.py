import os
import numpy as np
from uncertainties import ufloat
import json

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)


with open(f"{src_dir}/properties.json", 'r') as f:
  data = json.load(f)

# Constants
L, L0, tauf = data["L"], data["L0"], data["tauf"]
k, c, rho, h = data["k"], data["c"], data["rho"], data["h"]

Tmax, Troom, Tw = data["Tmax"], data["Troom"], data["Tw"] # tumour temperature in °C

SAR0, d, a, b, x0, beta = data["SAR0"], data["d"], data["a"], data["b"], data["x0"], data["beta"]

W1, W2, W3, alpha, eta = data["W1"], data["W2"], data["W3"], data["alpha"], data["eta"]

"coefficients a1, a2, a3, a4"

dT = Tmax - Troom

gamma = np.log(2)/(d - 10**(-2)*x0)

K = 15.0
delta = 0.1


a1 = round((L0**2/tauf)*((rho*c)/k), 7)
a2 = round(L0**2*c/k, 7)
a3 = round((rho*(L0**2)*beta*SAR0/(k*dT))*(np.exp(a*x0)), 7)
a4 = round(a*L0, 7)
a5 = round(L0*h/k, 7)




