import utils
import os
import numpy as np
from scipy.interpolate import interp1d


current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
project_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(project_dir, "tests")
logs_dir = os.path.join(tests_dir, "logs")

n=0
utils.get_properties(n)
# utils.gen_obsdata(n)
x = np.linspace(0, 1, 100)

a = utils.gen_obsdata(n)


# print(g[0])
print(utils.f3(x))
# print(instants)