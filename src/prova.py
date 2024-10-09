import numpy as np
import utils as uu
import os
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)


b = np.linspace(0.20675, 0.2435, num=8).round(5)

print(b)