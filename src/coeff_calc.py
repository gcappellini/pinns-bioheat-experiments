import os
import numpy as np
from uncertainties import ufloat

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

"coefficients a1, a2, a3, a4"

L0 = 15/100          # 5 cm
tauf = 30*60

k = 0.6
rho = 1000
c = 4180

Ta = 22
Tmax = 36
dT = Tmax - Ta

h = 200

d = 0.03

W = .3
P0 = 50.0
beta = 1.8e+03

z0 = 0.004
gamma = np.log(2)/(d - 10**(-2)*z0)

rho_wp = ufloat(1100.0, 100)
c_wp = 4180
k_wp = ufloat(0.55, 0.05)

C1 = (L0**2/(47*60))*(rho_wp * c_wp)/k_wp
C2 = (L0**2)*c/k_wp

a1 = round((L0**2/tauf)*((rho*c)/k), 7)
a2 = round(L0**2*(c * W)/k, 7)
a3 = round((L0**2)*beta*P0/(k*dT*(P0**(gamma*z0))), 7)
a4 = -2.0
a5 = round(L0*h, 7)
a6 = round(L0*gamma, 7)

lam = 10.0
K = 15.0
upsilon = 5.0

print(a1, a2, a3, a4, a5, a6, lam, K, upsilon)

# print(C1, C2)

hh = 1/((1/5000)+(0.5/(1000*0.2))+(0.3/(1000*0.2)))
print(hh)

