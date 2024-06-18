import utils
import os
import numpy as np

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
measurements_dir = os.path.join(src_dir, "measurements")

data = np.loadtxt(f"{measurements_dir}/phantom/20240403.txt")

print(data.shape)

