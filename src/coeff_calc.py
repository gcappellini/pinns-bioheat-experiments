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

omegab = 2.22e-03
tauf = 1800
Ta = 37
Tmax = 45
dT = Tmax - Ta

h = 300

# a1 = (tauf/(L0**2))*(keff/(rho*c))
# a2 = tauf*rhob*omegab*cb/(rho*c)
# a3 = tauf/(dT*rho*c)

a1 = (L0**2/tauf)*((rho*c)/k)
a2 = (cb * L0**2)/k
a3 = L0**2/(k*dT)
a5 = L0*h


print(f"a1:{a1}, a2:{a2}, a3:{a3}, a5:{a5}" )


