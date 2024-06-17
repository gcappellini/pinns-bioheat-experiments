import utils
import os
import numpy as np


current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
project_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(project_dir, "tests")
logs_dir = os.path.join(tests_dir, "logs")
# name = "mm9obs_test0_var0.6"
# a = np.load(f"{logs_dir}/{name}/weights_lambda_200.npy")

# # print(a[1:].shape)
# print(a[-3])


n_obs=9

for j in range(n_obs):
    print(j)