import utils_meas as utils
import os
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import coeff_calc as cc
from uncertainties import ufloat

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)


L0 = 10/100          

k_wallp = 0.6
k_plexi = 0.187
rho_water = 1000
rho_plexi = 1050
rho_wallp = 1000
c_water = 4181

tauf = 1800
Ta = 37
Tmax = 45
dT = Tmax - Ta

h = 300

d = 3.1/100

W = 0.45
P0 = 1e+05


a1 = round((L0**2/tauf)*((rho*c)/k), 7)
a2 = round(W*(cb * L0**2)/k, 7)
a3 = round(P0 * L0**2/(k*dT), 7)
a4 = round(0.7, 7)
a5 = round(L0*h, 7)
a6 = round(L0/d, 7)
