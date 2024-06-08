import numpy as np
import os



current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)




def gen_testdata(n):
    data = np.loadtxt(f"{src_dir}/simulations/file{n}.txt")
    x, t, exact = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:].T
    X = np.vstack((x, t)).T
    y = exact.flatten()[:, None]
    return X, y

def gen_obsdata(n):
    g = np.hstack((gen_testdata(n)))
    obs_data = g[(g[:, 0]== 0.0) | (g[:, 0]== 1.0)]
    return obs_data

# obs_mask = (e[:, 0] == 0) | (coords[:, 1] == 1.0)
# ic_mask = (coords[:, 1] == 0)