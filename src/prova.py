import utils_meas as utils
import os
import pandas as pd
import datetime
import numpy as np

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

b, c, a = utils.import_testdata("measurements/vessel/1")
print(b.shape, a.shape)