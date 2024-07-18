import os
import numpy as np
import matplotlib.pyplot as plt

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

"coefficients a1, a2, a3, a4"

L0 = 5/100          # 5 cm

k = 5.0
rhob = 1043
rho = 1050
c = 3639
cb = 3825

tauf = 1800
Ta = 37
Tmax = 45
dT = Tmax - Ta

h = 300

d = 0.03

W = 0.45
P0 = 1e+05

# a1 = (tauf/(L0**2))*(keff/(rho*c))
# a2 = tauf*rhob*omegab*cb/(rho*c)
# a3 = tauf/(dT*rho*c)

a1 = round((L0**2/tauf)*((rho*c)/k), 7)
a2 = round(W*(cb * L0**2)/k, 7)
a3 = round(P0 * L0**2/(k*dT), 7)
a4 = round(0.7, 7)
a5 = round(L0*h, 7)
a6 = round(L0/d, 7)

# print(f"a1:{a1}, a2:{a2}, a3:{a3}, a4:{a4}, a5:{a5} , a6:{a6}" )


