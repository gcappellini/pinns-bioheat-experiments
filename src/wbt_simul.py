import numpy as np
import matplotlib.pyplot as plt
import os

# Directories Setup
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
conf_dir = os.path.join(src_dir, "configs")
gt_dir = os.path.join(git_dir, "gt")
os.makedirs(tests_dir, exist_ok=True)

output_dir = f"{tests_dir}/cooling_simulation_wbt"

# Define the function u(x, t)
def u(x, t):
    return (3 * (x - 2)**2 + 6 * t) * np.exp(-t - t**2 - (t**3) / 3)/12
# Define the perfusion function q(t)
def q(t):
    return 1 + 2 * t + t**2


# Generate space (x) and time (t) values
x = np.linspace(0, 1, 100)
t = np.linspace(0, 1, 100)
x, t = np.meshgrid(x, t)

# Compute u(x, t)
u_values = u(x, t)

# # Compute q(t) values
# q_values = q(t[:, 0])  # Use the first column of t since t is uniform across rows

# # Create the scatter plot
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(x, t, c=u_values, cmap='viridis', marker='o')
# plt.colorbar(scatter, label='u(x, t)')
# plt.xlabel('Space (x)')
# plt.ylabel('Time (t)')
# plt.title('Adimensional Temperature u(x, t)')
# plt.show()



# # Plot q(t)
# plt.figure(figsize=(8, 6))
# plt.plot(t[:, 0], q_values, label='q(t)', color='blue')
# plt.xlabel('Time (t)')
# plt.ylabel('Perfusion q(t)')
# plt.title('Perfusion q(t) over Time')
# plt.legend()
# plt.grid()
# plt.show()