import numpy as np
import matplotlib.pyplot as plt
import os
import utils

current_file = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file)

K, L_0, h = 100, 0.15, 369.0
utils.set_K(K)
exper = "BX2"
t = 0.98
e = utils.obs_data(exper)
xex, mem = utils.meas_data(exper)
x_right = np.concatenate((xex, mem), axis=1)
# x_ini = e[e[:, -1]==t]
# theta_true = x_right[x_right[:, 1]==t]



# Find the index of the closest value to t in the array
indices_closest_t = np.where(np.abs(e[:, -1] - t) == np.min(np.abs(e[:, -1] - t)))[0]

# Get the component with the closest value to t
x_ini = e[indices_closest_t]
# Find the index of the closest value to t in the second column of x_right
indices_closest_x_right = np.where(np.abs(x_right[:, 1] - t) == np.min(np.abs(x_right[:, 1] - t)))[0]

# Get the component with the closest value to t
theta_true = x_right[indices_closest_x_right]

x = np.linspace(0, 1, num=100)
x_needle = np.linspace(0, 1, num=7)

x_pred = np.vstack((x, np.full_like(x, x_ini[0, 1]), np.full_like(x, x_ini[0, 2]), np.full_like(x, x_ini[0, 3]), np.full_like(x, t))).T

a = utils.create_observer(h)
b = utils.restore_model(a, f"obs_{h}")
theta_pred = b.predict(x_pred).reshape(x.shape)
print(x.shape, theta_pred.shape)

# Plotting
# plt.figure(figsize=(8, 6))
# plt.imshow(temperature, extent=[x.min(), x.max(), t.min(), t.max()], aspect='auto', cmap='viridis')
plt.plot(x_needle, theta_true[:, -1], linewidth=0, marker='x', color='r', label='measured')
plt.plot(x, theta_pred, label='predicted')
# plt.colorbar(label='Temperature')
plt.title(f'{exper}, t={t}')
plt.legend()
plt.xlabel('Z')
plt.ylabel(r'$\tau$')
plt.savefig(f'{script_directory}/prova_2.png')
plt.show()

