import utils
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

mu, sigma = 5, 2.5 # mean and standard deviation
s = np.random.normal(mu, sigma, 8).round(2)

print(s)
