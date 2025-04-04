% Define the PDE coefficients for pdepe
function [c, f, s] = pde_coeffs(x, t, u, dudx)
    q = 1 + 2 * t + t^2; % Perfusion term
    c = 1; % Coefficient of du/dt
    f = dudx; % Coefficient of spatial derivative
    s = -q * u; % Source term
end

% Define the initial condition
function u0 = initial_condition(x)
    u0 = 0.25 * (x - 2)^2; % Initial condition at t = 0
end

% Define the boundary conditions
function [pl, ql, pr, qr] = boundary_conditions(xl, ul, xr, ur, t)
    pl = -0.5*(xr-2)*exp(-t-t.^2-(t.^3)/3); % Left boundary condition
    ql = 1; % No flux at left boundary
    pr = ur - (0.25 + 0.5 * t) * exp(-t - t^2 - (t^3) / 3); % Right boundary condition
    qr = 0; % No flux at right boundary
end

% Define the spatial and temporal domains
x = linspace(0, 1, 50); % Spatial domain
t = linspace(0, 1, 50); % Temporal domain

% Solve the PDE using pdepe
m = 0; % Symmetry parameter (0 for slab geometry)
sol = pdepe(m, @pde_coeffs, @initial_condition, @boundary_conditions, x, t);

% Extract the solution
theta_sol = sol;

% Plot the solution
figure;
surf(x, t, theta_sol);
xlabel('Spatial coordinate x');
ylabel('Time t');
zlabel('Solution \theta(x, t)');
title('Solution of the PDE using pdepe');

% Define the exact solution
[X, T] = meshgrid(x, t);
theta_exact = (0.25 * (X - 2).^2 + 0.5 * T) .* exp(-T - T.^2 - (T.^3) / 3);

% Plot the exact solution
figure;
surf(x, t, theta_exact);
xlabel('Spatial coordinate x');
ylabel('Time t');
zlabel('Exact Solution \theta(x, t)');
title('Exact Solution of the PDE');


% Compute the error between the numerical and exact solutions
error = abs(theta_sol - theta_exact);

% Plot the error
figure;
surf(x, t, error);
xlabel('Spatial coordinate x');
ylabel('Time t');
zlabel('Error |Numerical - Exact|');
title('Error between Numerical and Exact Solutions');
