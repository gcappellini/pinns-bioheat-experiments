import numpy as np
import utils as uu
import os
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

a = uu.load_from_pickle(f"{src_dir}/data/vessel/cooling_meas_1.pkl")
print(a)