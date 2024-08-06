function main
    % Main function to solve PDE using pdepe

    % Define global variables for coefficients
    global a1 a2 a3 a4 W1 W2 W3 R2;

    % Load properties from JSON file
    [a1, a2, a3, a4, W1, W2, W3, R2] = loadProperties('properties.json');

    m = 0; % Symmetry for PDE (Cartesian coordinates)
    x = linspace(-R2, R2, 101); % Define spatial domain
    t = linspace(0, 1, 101); % Define time domain

    % Solve PDE
    sol = pdepe(m, @OneDimBHpde, @OneDimBHic, @OneDimBHbc, x, t);

    % Extract the solution components
    u1 = sol(:,:,1); % Solution of system 1
    u2 = sol(:,:,2); % Solution of system 2
    u3 = sol(:,:,3); % Solution of system 3

    % Print Solution PDE to a file
    fileID = fopen('output_pbhe.txt', 'w');

    for i = 1:101
        fprintf(fileID, '%12.8f %12.8f %12.8f %12.8f\n', ...
            x(i), u1(end, i), u2(end, i), u3(end, i));
    end

    fclose(fileID);

    % Plotting the solution for visualization
    figure;
    plot(x, u1(end, :), '-o', x, u2(end, :), '-s', x, u3(end, :), '-d');
    xlabel('Spatial Domain (x)');
    ylabel('Theta');
    title('Solution of PBHE at Final Time Step');
    legend('u1', 'u2', 'u3');
    grid on;
end

function [a1, a2, a3, a4, W1, W2, W3, R2] = loadProperties(filename)
    % Check if filename is provided
    if nargin < 1
        filename = 'properties.json'; % Default filename
    end

    % Read JSON file
    try
        jsonData = fileread(filename);  % Read the JSON file as a string
    catch
        error('Error reading the JSON file. Make sure the file exists and the path is correct.');
    end

    % Parse JSON data
    try
        data = jsondecode(jsonData);    % Decode JSON data into a MATLAB struct
    catch
        error('Error parsing JSON file. Ensure it is correctly formatted.');
    end

    % Extract parameters from the struct
    try
        R2 = data.R2;
        k = data.k;
        rho_fl = data.rho_fl;
        c_fl = data.c_fl;
        omega_fl = data.omega_fl;
        W1 = data.W1;
        W2 = data.W2;
        W3 = data.W3;
        a4 = data.a4;
    catch
        error('Error accessing JSON fields. Ensure JSON contains required fields.');
    end

    % Displaying the loaded parameters for debugging
    fprintf('Loaded R2: %f\n', R2);
    fprintf('Loaded k: %f\n', k);
    fprintf('Loaded rho_fl: %f\n', rho_fl);
    fprintf('Loaded c_fl: %f\n', c_fl);
    fprintf('Loaded omega_fl: %f\n', omega_fl);
    fprintf('Loaded W1: %f\n', W1);
    fprintf('Loaded W2: %f\n', W2);
    fprintf('Loaded W3: %f\n', W3);
    fprintf('Loaded a4: %f\n', a4);

    % Compute constants a1, a2, and a3
    a1 = rho_fl * c_fl / 1e+50;
    a2 = k / (2 * R2);
    a3 = rho_fl * c_fl * omega_fl;
end

function [c, f, s] = OneDimBHpde(x, t, u, dudx)
    % Define the coefficients of the PDE
    global a1 a2 a3 a4 W1 W2 W3;
    c = [a1; a1; a1]; % Coefficient c in PDE
    f = a2 * dudx; % Flux term with scaling
    % Source term with varying coefficients
    s = [-W1 * a3 * u(1)+a4;
         -W2 * a3 * u(2)+a4;
         -W3 * a3 * u(3)+a4];
end

function u0 = OneDimBHic(x)
    % Initial conditions for the PDE
    u0 = ones(3, 1); % Example: all ones for initial conditions
end

function [pl, ql, pr, qr] = OneDimBHbc(xl, ul, xr, ur, t)
    % Left boundary conditions (Dirichlet: u = 1)
    pl = [ul(1) - 1; ul(2) - 1; ul(3) - 1]; % pl = ul - desired_value
    ql = zeros(3, 1); % ql = 0 for Dirichlet

    % Right boundary conditions (Dirichlet: u = 1)
    pr = ur - 1; % pr = ur - desired_value
    qr = zeros(3, 1); % qr = 0 for Dirichlet
end