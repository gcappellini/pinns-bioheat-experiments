import os
import numpy as np
import matplotlib.pyplot as plt

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

"coefficients a1, a2, a3, a4"

L0 = 5/100

keff = 5.0
rhob = 1043
rho = 1050
c = 3639
cb = 3825

omegab = 2.22e-03
tauf = 1800
Ta = 37
Tmax = 45
dT = Tmax - Ta

a1 = (tauf/(L0**2))*(keff/(rho*c))
a2 = tauf*rhob*omegab*cb/(rho*c)
a3 = tauf/(dT*rho*c)
print(a1, a2, a3)
q0 = 16
beta = 15
a4 = q0/(4*dT)
a5 = beta/dT
print(a4, a5)



