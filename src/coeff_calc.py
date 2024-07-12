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

d = 0.3

W = 0.45
obs = np.array([1, 2, 3, 5, 6, 8, 9, 10])

# W_obs = np.array(W*[1, 2, 3, 5, 6, 8, 9, 10]).round(3)
W_obs = np.dot(W, obs)

# a1 = (tauf/(L0**2))*(keff/(rho*c))
# a2 = tauf*rhob*omegab*cb/(rho*c)
# a3 = tauf/(dT*rho*c)

a1 = (L0**2/tauf)*((rho*c)/k)
a2 = (cb * L0**2)/k
a3 = L0**2/(k*dT)
a4 = 0.7
a5 = L0*h
a6 = L0/d

print(len(obs))
print(f"a1:{round(a1, 7)}, a2:{round(a2, 7)}, a3:{round(a3, 7)}, a4:{round(a4, 7)}, a5:{round(a5, 7)} , a6:{round(a6, 7)}" )


