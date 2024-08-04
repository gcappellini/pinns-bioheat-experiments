import matlab.engine
import os
import ratio_along_vessel as rr
import matplotlib.pyplot as plt
import numpy as np

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)

# eng = matlab.engine.start_matlab()
# eng.cd(src_dir, nargout=0)
# eng.simple_script(nargout=0)
# eng.quit()

R2 = 4.0/100
# Grid settings
xgr = 30/100  # x-axis (length along vessel in cm)
r = np.linspace(-R2/2, R2/2, 100)  # radial axis (from -1 cm to 1 cm)

R1, k, dx =  1/1000, 0.6, 0.5/100
v = 1.5/100
# Calculate normalized temperature difference
df1 = rr.vessel_temperature_distribution(v, R1)
tt = rr.temperature_distribution(v, R1, xgr)
x_vals, t_blood_vals, t_wall_vals, q = df1[0], df1[1], df1[2], df1[3]
T_w_normalized = rr.scale_t(t_wall_vals)
T_b_normalized = rr.scale_t(t_wall_vals)

x_index = np.argmin(np.abs(x_vals - xgr))
T_tissue = t_wall_vals[x_index] + q[x_index] * ((np.log(np.abs(r) / R1) / (2 * np.pi * k * dx)))
T = np.where(np.abs(r)<=R1, t_wall_vals[x_index], T_tissue)
T_normalized = rr.scale_t(T)


# Plot the 3D temperature distribution
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.plot(r*100, T_normalized, color='black', linewidth=1.5)


# Customize the plot
ax.set_xlabel('Radial Distance, $r$ (cm)', fontsize=12)
ax.set_ylabel(r'$\frac{T_{\text{tissue}}(x, r) - T_{\text{fluid}}(0)}{T_{\text{tumour}} - T_{\text{fluid}}(0)}$', fontsize=12)
ax.set_title(f'Temperature Distribution Along Radius, x={xgr}', fontsize=14)

# ax.legend()


plt.show()

