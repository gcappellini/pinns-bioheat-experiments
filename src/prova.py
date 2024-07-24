import utils_meas as utils
import os
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import coeff_calc as cc
from uncertainties import ufloat
from scipy.interpolate import interp1d

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

L0 = 15/100 
d = 0.03
a6 = round(L0/d, 7)

z0 = 0.004
PD = 2*d


c = np.log(2)/(PD - z0*1e-2)
print(a6, c)