import numpy as np
import os
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)


a = np.loadtxt(f"{current_dir}/20240403.txt", skiprows=0)
print(a[:, 0])