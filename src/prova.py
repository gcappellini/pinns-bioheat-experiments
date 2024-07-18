import utils_meas as utils
import os
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import coeff_calc as cc

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)



a = utils.import_obsdata("measurements/vessel/3")

print(a)
