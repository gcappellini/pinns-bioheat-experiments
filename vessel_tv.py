import numpy as np
import os
import matplotlib.pyplot as plt

current_file = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file)
output_dir = f"{script_directory}/vessel"
os.makedirs(output_dir, exist_ok=True)

def xeq(v, r1, yv, kbl, keff, rho_bl, cbl):
    return (1/2) * rho_bl * cbl * v * r1**2 * (1/2 + (kbl/keff) * np.log(yv/r1)) / kbl

def Tv(x, xeq, Tbl):
    return Tbl * (1 - np.exp(-x/xeq))

# Define the intervals for r1 and v
r1_values = np.linspace(0.01, 0.1, 100)
v_values = np.linspace(0.05, 0.1, 100)

# Define other parameters
Tbl =  22.0 # You need to specify Tbl
kbl =  0.52 # You need to specify kbl
keff =  5.0 # You need to specify keff
rho_bl =  1043.0 # You need to specify rho_bl
cbl =  3825.0 # You need to specify cbl
yit = 0.05
yv =  (3/13)*yit # total height is 13 cm. vessel should flow at 3 cm from top
x =  0.5 # You need to specify the value of x

# Calculate Tv for each combination of r1 and v
Tv_values = np.zeros((len(r1_values), len(v_values)))

for i, r1 in enumerate(r1_values):
    for j, v in enumerate(v_values):
        xeq_value = xeq(v, r1, yv, kbl, keff, rho_bl, cbl)
        Tv_values[i, j] = Tv(x, xeq_value, Tbl)

# Plot Tv versus r1 and v
plt.figure(figsize=(10, 6))
plt.contourf(v_values, r1_values, Tv_values.T, cmap='viridis')
plt.colorbar(label='Tv')
plt.xlabel('v')
plt.ylabel('r1')
plt.title('Tv versus r1 and v')
plt.grid(True)
plt.savefig(f'{output_dir}/Tv_vs_r1_and_v.png')
plt.show()