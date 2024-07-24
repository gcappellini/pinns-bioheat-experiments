import os
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

"coefficients a1, a2, a3, a4"

L0 = 15/100          # 5 cm
tauf = 47*60

k = 0.55
kw = 0.598
rhob = 1043
rho = 1110
c = 4180
cw = 4180


Ta = 37
Tmax = 45
dT = Tmax - Ta

h = 300

d = 0.03

W = 1.2
P0 = 0

rho_wp = ufloat(1100.0, 100)
c_wp = 4180
k_wp = ufloat(0.55, 0.05)
C1 = (L0**2/(47*60))*(rho_wp * c_wp)/k_wp
C2 = (L0**2)*cw/k_wp

a1 = round((L0**2/tauf)*((rho*c)/k), 7)
a2 = round(W*(cw * L0**2)/k, 7)
a3 = round(P0 * L0**2/(k*dT), 7)
a5 = round(L0*h, 7)
a6 = round(L0/d, 7)

print(f"a1:{a1}, a2:{a2}, a3:{a3}, a4:{a4}, a5:{a5} , a6:{a6}" )

# print(C1, C2)

