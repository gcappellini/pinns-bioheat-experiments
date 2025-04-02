import torch
import torch.nn as nn
import torch.optim as optim

# GRU-based Surrogate Model
class PhysicsInformedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(PhysicsInformedGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h, _ = self.gru(x)
        return self.fc(h)

# Fully Connected Neural Network (FCNN) for T(x, t)
class TemperatureNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=1):
        super(TemperatureNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, t):
        xt = torch.cat((x, t), dim=-1)  # Concatenate x and t as inputs
        return self.net(xt)

# Autodiff-based Physics Loss
def physics_loss(T_nn, x, t, k, omega, rho, c_p, rho_b, c_b, T_b, Q_m, Q_ext):
    """
    Compute the residual of the Pennes Bioheat equation using autograd.
    """
    T_pred = T_nn(x, t)
    T_pred.requires_grad_(True)  # Ensure autograd tracks computation

    # Compute dT/dt using autodiff
    dT_dt = torch.autograd.grad(T_pred, t, grad_outputs=torch.ones_like(T_pred), create_graph=True)[0]

    # Compute d²T/dx² using autodiff
    dT_dx = torch.autograd.grad(T_pred, x, grad_outputs=torch.ones_like(T_pred), create_graph=True)[0]
    d2T_dx2 = torch.autograd.grad(dT_dx, x, grad_outputs=torch.ones_like(dT_dx), create_graph=True)[0]

    # Compute bioheat equation residual
    perfusion_term = omega * rho_b * c_b * (T_b - T_pred)
    bioheat_residual = rho * c_p * dT_dt - k * d2T_dx2 - perfusion_term - Q_m - Q_ext

    return torch.mean(bioheat_residual**2)  # Penalize equation violation

# Training Loop
def train_model(gru_model, nn_model, train_data, target, x, t, k, omega, rho, c_p, rho_b, c_b, T_b, Q_m, Q_ext, epochs=100, lr=0.001):
    optimizer_gru = optim.Adam(gru_model.parameters(), lr=lr)
    optimizer_nn = optim.Adam(nn_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer_gru.zero_grad()
        optimizer_nn.zero_grad()
        
        output = gru_model(train_data)
        
        # Standard loss (data-driven)
        data_loss = criterion(output, target)
        
        # Physics-informed loss
        phys_loss = physics_loss(nn_model, x, t, k, omega, rho, c_p, rho_b, c_b, T_b, Q_m, Q_ext)
        
        # Total loss
        loss = data_loss + phys_loss
        
        loss.backward()
        optimizer_gru.step()
        optimizer_nn.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Example Usage
input_size = 1  # Temperature at each time step
hidden_size = 32
output_size = 1
num_layers = 1

gru_model = PhysicsInformedGRU(input_size, hidden_size, output_size, num_layers)
nn_model = TemperatureNN()

# Fake training data (modify for real data)
train_data = torch.randn(100, 10, 1, requires_grad=True)  # Batch of 100 sequences, each of length 10
target = torch.randn(100, 10, 1)      # Target temperature values

# Time and spatial grid (assume 1D case)
t = torch.linspace(0, 1, 100, requires_grad=True).view(-1, 1)  # Time steps
x = torch.linspace(0, 1, 100, requires_grad=True).view(-1, 1)  # Spatial steps

# Physics parameters (modify for real case)
k = 0.5    # Thermal conductivity (W/m·K)
omega = 0.01  # Blood perfusion rate
rho = 1000  # Tissue density (kg/m³)
c_p = 3600  # Specific heat (J/kg·K)
rho_b = 1060  # Blood density (kg/m³)
c_b = 3770  # Blood specific heat (J/kg·K)
T_b = 310  # Blood temperature (K)
Q_m = 500  # Metabolic heat (W/m³)
Q_ext = 0  # External heat (W/m³)

train_model(gru_model, nn_model, train_data, target, x, t, k, omega, rho, c_p, rho_b, c_b, T_b, Q_m, Q_ext)