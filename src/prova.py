import matlab.engine
import os
import matplotlib.pyplot as plt
import numpy as np

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)

# eng = matlab.engine.start_matlab()
# eng.cd(src_dir, nargout=0)
# eng.simple_script(nargout=0)
# eng.quit()


# Constants
a = 1.0  # You can change this value
b = .04  # You can change this value
yb = .80 # Boundary condition value

# Derived constant
lambda_ = np.sqrt(b / a)

# Constants for the solution
C1 = yb * (1 - np.exp(-lambda_)) / (2 * np.sinh(lambda_))
C2 = yb * (np.exp(lambda_) + 1) / (2 * np.sinh(lambda_))

# Function for the solution
def y(x):
    return C1 * np.exp(lambda_ * x) + C2 * np.exp(-lambda_ * x)

# Define the domain
x_values = np.linspace(0, 1, 500)

# Compute y values
y_values = y(x_values)

# Plot the solution
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label=r'$y(x) = C_1 e^{\lambda x} + C_2 e^{-\lambda x}$', color='blue')
plt.axhline(y=yb, color='gray', linestyle='--', linewidth=1, label=r'$y(0) = y(1) = y_b$')
plt.scatter([0, 1], [yb, yb], color='red', zorder=5)
plt.title('Solution of the Differential Equation')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.grid(True)
plt.legend()
# plt.xlim(0, 1)
# plt.ylim(0.9 * yb, 1.1 * yb)
plt.show()
