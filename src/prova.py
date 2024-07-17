import utils_meas as utils
import os
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)



n = "measurements/vessel/1"  # Example argument
Xobs = utils.import_obsdata(n)
print(Xobs)