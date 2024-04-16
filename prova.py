import numpy as np
import matplotlib.pyplot as plt
import os

current_file = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file)

# Generate some sample data
x = np.linspace(0, 10, 100)  # x coordinates
t = np.linspace(0, 5, 50)     # t coordinates
X, T = np.meshgrid(x, t)      # Create a meshgrid of x and t
temperature_1d = np.sin(X.flatten()) * np.cos(T.flatten())  # Example temperature data
la = len(x)
le = len(t)
temperature = temperature_1d.reshape((le, la))

# Plotting
plt.figure(figsize=(8, 6))
plt.imshow(temperature, extent=[x.min(), x.max(), t.min(), t.max()], aspect='auto', cmap='viridis')
plt.colorbar(label='Temperature')
plt.title('Temperature Distribution')
plt.xlabel('x')
plt.ylabel('t')
plt.savefig(f'{script_directory}/prova.png')
plt.show()