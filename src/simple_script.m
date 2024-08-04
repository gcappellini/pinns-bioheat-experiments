function [R2, k, rho_fl, c_fl, omega_fl, e0, e1, e2] = loadProperties(filename)
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
    catch
        error('Error accessing JSON fields. Ensure JSON contains required fields.');
    end

    % Displaying the loaded parameters for debugging
    fprintf('Loaded R2: %f\n', R2);
    fprintf('Loaded k: %f\n', k);
    fprintf('Loaded rho_fl: %f\n', rho_fl);
    fprintf('Loaded c_fl: %f\n', c_fl);
    fprintf('Loaded omega_fl: %f\n', omega_fl);
    
    % Compute constants e1 and e2
    e0 = rho_fl * c_fl /1e05;
    e1 = k / (2 * R2);
    e2 = rho_fl * c_fl * omega_fl;
end

function sol = OneDimBH
    % Load properties from JSON file
    [R2, k, rho_fl, c_fl, omega_fl, e0, e1, e2] = loadProperties('properties.json');
    
    m = 0; % Symmetry for PDE
    t = linspace(0,1,101);
    x = linspace(0,1,101); % Define spatial domain
    
    % Solve PDE
    sol = pdepe(m,@OneDimBHpde,@OneDimBHic,@OneDimBHbc,x);
    
    % Extract the first solution component as u.
    u1 = sol(:,:,1); % Solution of system 1
    u2 = sol(:,:,2); % Solution of system 2
    u3 = sol(:,:,3); % Solution of system 3
    u4 = sol(:,:,4); % Solution of system 4
    u5 = sol(:,:,5); % Solution of system 5
    u6 = sol(:,:,6); % Solution of system 6
    u7 = sol(:,:,7); % Solution of system 7
    u8 = sol(:,:,8); % Solution of system 8

    % Print Solution PDE to a file
    fileID = fopen('output_pbhe.txt','w');
    
    for i = 1:101
        fprintf(fileID,'%6.2f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f\n', ...
            x(i, 101), u1(i, 101), u2(i, 101), u3(i, 101), u4(i, 101), u5(i, 101), u6(i, 101), u7(i, 101), u8(i, 101));
    end
    
    fclose(fileID);
    
    % Nested functions
    function [c,f,s] = OneDimBHpde(x,u,dudx)
        % Define the coefficients of the PDE
        c = e0 * ones(8, 1); % 8 zeroes for c
        f = e1 * ones(8, 1) * dudx; % Scaling for f

        % Source term with varying coefficients
        s = [-1*e2*u(1); 
            -2*e2*u(2); 
            -3*e2*u(3); 
            -4*e2*u(4); 
            -5*e2*u(5); 
            -6*e2*u(6); 
            -7*e2*u(7); 
            -8*e2*u(8)];
    end

    function u0 = OneDimBHic(x)
        % Initial conditions for the PDE
        u0 = ones(8, 1); % Example: all ones for initial conditions
    end

    function [pl,ql,pr,qr] = OneDimBHbc(xl,ul,xr,ur)
        % Left boundary conditions
        pl = ul; 
        ql = ones(8, 1);

        % Right boundary conditions
        pr = ur; 
        qr = ones(8, 1);
    end
end

% Call the OneDimBH function
sol = OneDimBH;