import utils_meas as utils
import os
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)


# Example vectors
e = np.random.rand(48, 2)         # Shape (48, 2)
theta_true = np.random.rand(48)   # Shape (48,)
theta_pred = np.random.rand(48)   # Shape (48,)

# Reshape theta_true and theta_pred to (48, 1)
theta_true = theta_true.reshape(48, 1)
theta_pred = theta_pred.reshape(48, 1)

print(e.shape, theta_true.shape, theta_pred.shape)

# Combine vectors
tot = np.hstack((e, theta_true, theta_pred))

print(tot.shape)  # Should print (48, 4)