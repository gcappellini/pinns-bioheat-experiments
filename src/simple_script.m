function main
    % Main function to solve PDE using pdepe

    % Define global variables for coefficients
    global a1 a2 a3 a4 a5 W1 W2 W3 theta1 thetaw;

    % Load properties from JSON file
    [a1, a2, a3, a4, a5, W1, W2, W3, theta1, thetaw] = loadProperties('properties.json');

    m = 0; % Symmetry for PDE (Cartesian coordinates)
    x = linspace(0, 1, 101); % Define spatial domain
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

function [a1, a2, a3, a4, a5, W1, W2, W3, theta1, thetaw] = loadProperties(filename)
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
        L0 = data.L0;
        tauf = data.tauf;
        k = data.k;
        rho = data.rho;
        c = data.c;
        beta = data.beta;
        P0 = data.P0;
        d = data.d;
        a = data.a;
        x0 = data.x0;
        t_tis = data.Ttis;
        t_w = data.Tw;
        h = data.h;
        dT = data.dT;
        W1 = data.W1;
        W2 = data.W2;
        W3 = data.W3;
    catch
        error('Error accessing JSON fields. Ensure JSON contains required fields.');
    end

    % Displaying the loaded parameters for debugging
    fprintf('Loaded L0: %f\n', L0);
    fprintf('Loaded tauf: %f\n', tauf);
    fprintf('Loaded k: %f\n', k);
    fprintf('Loaded rho: %f\n', rho);
    fprintf('Loaded c: %f\n', c);
    fprintf('Loaded P0: %f\n', P0);
    fprintf('Loaded d: %f\n', d);
    fprintf('Loaded x0: %f\n', x0);
    fprintf('Loaded dT: %f\n', dT);
    fprintf('Loaded W1: %f\n', W1);
    fprintf('Loaded W2: %f\n', W2);
    fprintf('Loaded W3: %f\n', W3);
    fprintf('Loaded h: %f\n', h);
    fprintf('Loaded t_tis: %f\n', t_tis);
    fprintf('Loaded t_w: %f\n', t_w);

    % Compute constants a1, a2, a3, and a4
    a1 = (L0^2/tauf)*(rho*c/k);
    a2 = L0^2*c/k;
    a3 = (L0^2/(k*dT))*3e+03*beta*(P0)*exp(a*x0);
    a4 = a*L0;
    % a3 = (L0^2/dT)*P0;
    % a4 = L0/d;
    a5 = (h*L0)/k;
    thetaw = (t_w - t_tis)/dT;
    theta1 = 0;
end

function [c, f, s] = OneDimBHpde(x, t, u, dudx)
    % Define the coefficients of the PDE
    global a1 a2 a3 a4 W1 W2 W3 ;
    c = [a1; a1; a1]; % Coefficient c in PDE
    f = 1 * dudx; % Flux term with scaling
    % Source term with varying coefficients
    s = [-W1 * a2 * u(1)+a3*exp(-a4*x);
         -W2 * a2 * u(2)+a3*exp(-a4*x);
         -W3 * a2 * u(3)+a3*exp(-a4*x)];
end

function u0 = OneDimBHic(x)
    % Initial conditions for the PDE
    u0 = ones(3, 1); % Example: all ones for initial conditions
end

function [pl, ql, pr, qr] = OneDimBHbc(xl, ul, xr, ur, t)
    global a5 theta1 thetaw;

    % Robin boundary condition at x=0
    pl = a5*(thetaw - ul); % Represents (h*theta - h*theta_w)
    ql = ones(3, 1);

    % Right boundary conditions (Dirichlet: u = 1)
    pr = ur - theta1; % pr = ur - desired_value
    qr = zeros(3, 1); % qr = 0 for Dirichlet
end