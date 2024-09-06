function main
    % Main function to solve PDE using pdepe

    % Define global variables for coefficients
    global a1 a2 a3 a4 a5 delta alpha lambda W0 W1 W2 W3 W4 W5 W6 W7 theta1 thetaw om0 om1 om2  om3 om4  om5 om6 om7 ;

    % Load properties from JSON file
    [a1, a2, a3, a4, a5, delta, alpha, lambda, W0, W1, W2, W3, W4, W5, W6, W7, theta1, thetaw] = loadProperties('properties.json');

    m = 0; % Symmetry for PDE (Cartesian coordinates)
    x = linspace(0, 1, 101); % Define spatial domain
    t = linspace(0, 1, 101); % Define time domain

    % Solve PDE
    sol = pdepe(m, @OneDimBHpde, @OneDimBHic, @OneDimBHbc, x, t);

    % Extract the solution components
    u1 = sol(:,:,1); %soluzione del sistema
    u2 = sol(:,:,2); %soluzione dell'osservatore 0
    u3 = sol(:,:,3); %soluzione dell'osservatore 1
    u4 = sol(:,:,4); %soluzione dell'osservatore 2
    u5 = sol(:,:,5); %soluzione dell'osservatore 3
    u6 = sol(:,:,6); %soluzione dell'osservatore 4
    u7 = sol(:,:,7); %soluzione dell'osservatore 5
    u8 = sol(:,:,8); %soluzione dell'osservatore 6
    u9 = sol(:,:,9); %soluzione dell'osservatore 7
    
    u10 = sol(:,:,10); %soluzione del peso 0
    u11 = sol(:,:,11); %soluzione del peso 1
    u12 = sol(:,:,12); %soluzione del peso 2
    u13 = sol(:,:,13); %soluzione del peso 3
    u14 = sol(:,:,14); %soluzione del peso 4
    u15 = sol(:,:,15); %soluzione del peso 5
    u16 = sol(:,:,16); %soluzione del peso 6
    u17 = sol(:,:,17); %soluzione del peso 7

    om0=0;
    om1=0;
    om2=0;
    om3=0;
    om4=0;
    om5=0;
    om6=0;
    om7=0;
    
    
    %multiple-model temperature estimation
    uav=u2.*u10+u3.*u11+u4.*u12+u5.*u13+u6.*u14+u7.*u15+u8.*u16+u9.*u17;

    % Print Solution PDE to a file
    fileID = fopen('output_pbhe.txt', 'w');

    for i = 1:101
        fprintf(fileID, '%12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f\n', ...
            x(i), u1(end, i), u2(end, i), u3(end, i), u4(end, i), u5(end, i), u6(end, i), u7(end, i), u8(end, i), u9(end, i), uav(end, i));
    end

    fclose(fileID);
    % Print Solution PDE to a file with time variation
    fileID_time = fopen('output_time_pbhe.txt', 'w');

    % Loop over time and spatial points to write data
    for j = 1:101  % Loop over each time step
        for i = 1:101  % Loop over each spatial point
            fprintf(fileID_time, '%12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f\n', ...
                x(i), t(j), u1(j, i), u2(j, i), u3(j, i), u4(j, i), u5(j, i), u6(j, i), u7(j, i), u8(j, i), u9(j, i), uav(j, i));
        end
        % Optional: Add a newline for clarity after each time step
        fprintf(fileID_time, '\n');
    end

    fclose(fileID_time);
end

function [a1, a2, a3, a4, a5, delta, alpha, lambda, W0, W1, W2, W3, W4, W5, W6, W7, theta1, thetaw] = loadProperties(filename)
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
        SAR0 = data.SAR0;
        d = data.d;
        a = data.a;
        x0 = data.x0;
        t_room = data.Troom;
        t_w = data.Tw;
        h = data.h;
        dT = data.dT;
        W0 = data.W0;
        W1 = data.W1;
        W2 = data.W2;
        W3 = data.W3;
        W4 = data.W4;
        W5 = data.W5;
        W6 = data.W6;
        W7 = data.W7;
        alpha = data.alpha;
        delta = data.delta;
        lambda = data.lambda;

    catch
        error('Error accessing JSON fields. Ensure JSON contains required fields.');
    end

    % Displaying the loaded parameters for debugging
    fprintf('Loaded L0: %f\n', L0);
    fprintf('Loaded tauf: %f\n', tauf);
    fprintf('Loaded k: %f\n', k);
    fprintf('Loaded rho: %f\n', rho);
    fprintf('Loaded c: %f\n', c);
    fprintf('Loaded SAR0: %f\n', SAR0);
    fprintf('Loaded d: %f\n', d);
    fprintf('Loaded x0: %f\n', x0);
    fprintf('Loaded dT: %f\n', dT);
    fprintf('Loaded W0: %f\n', W0);
    fprintf('Loaded W1: %f\n', W1);
    fprintf('Loaded W2: %f\n', W2);
    fprintf('Loaded W3: %f\n', W3);
    fprintf('Loaded W4: %f\n', W4);
    fprintf('Loaded W5: %f\n', W5);
    fprintf('Loaded W6: %f\n', W6);
    fprintf('Loaded W7: %f\n', W7);
    fprintf('Loaded h: %f\n', h);
    fprintf('Loaded t_tis: %f\n', t_room);
    fprintf('Loaded t_w: %f\n', t_w);

    % Compute constants a1, a2, a3, and a4
    a1 = (L0^2/tauf)*(rho*c/k);
    a2 = L0^2*c/k;
    a3 = (L0^2/(k*dT))*3e+03*beta*(SAR0)*exp(a*x0);
    a4 = a*L0;
    a5 = (h*L0)/k;
    thetaw = (t_w - t_room)/dT;
    theta1 = 0;
end

function [c, f, s] = OneDimBHpde(x, t, u, dudx)
    % Define the coefficients of the PDE
    global lambda a1 a2 a3 a4 W0 W1 W2 W3 W4 W5 W6 W7 om0 om1 om2 om3 om4 om5 om6 om7;
    t

    c = [a1; a1; a1; a1; a1; a1; a1; a1; a1; 1; 1; 1; 1; 1; 1; 1; 1];
    % f = [1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1].* dudx;
    f = 1.* dudx;

    den=u(10)*exp(-om0)+u(11)*exp(-om1)+u(12)*exp(-om2)+u(13)*exp(-om3)+...
        u(14)*exp(-om4)+u(15)*exp(-om5)+u(16)*exp(-om6)+u(17)*exp(-om7);

    s = [-W0*a2*u(1)+a3*exp(-a4*x); 
        -W0*a2*u(2)+a3*exp(-a4*x); 
        -W1*a2*u(3)+a3*exp(-a4*x); 
        -W2*a2*u(4)+a3*exp(-a4*x); 
        -W3*a2*u(5)+a3*exp(-a4*x); 
        -W4*a2*u(6)+a3*exp(-a4*x); 
        -W5*a2*u(7)+a3*exp(-a4*x); 
        -W6*a2*u(8)+a3*exp(-a4*x); 
        -W7*a2*u(9)+a3*exp(-a4*x); 
        -lambda*u(10)*(1-(exp(-om0)/den));
        -lambda*u(11)*(1-(exp(-om1)/den)); 
        -lambda*u(12)*(1-(exp(-om2)/den)); 
        -lambda*u(13)*(1-(exp(-om3)/den));
        -lambda*u(14)*(1-(exp(-om4)/den));
        -lambda*u(15)*(1-(exp(-om5)/den)); 
        -lambda*u(16)*(1-(exp(-om6)/den)); 
        -lambda*u(17)*(1-(exp(-om7)/den));
        ];
end

function u0 = OneDimBHic(x)
    global K delta a5

    y1_0 = 0;
    y2_0 = 0;
    y3_0 = 0;
    b1 = (a5*y3_0+(K-a5)*y2_0-(2+K)*delta)/(1+K);
    ic_obs = y1_0 + b1*x + delta*x^2;
    u0 = [0; ic_obs;  ic_obs; ic_obs; ic_obs; ic_obs; ic_obs; ic_obs; ic_obs; 1/8; 1/8; 1/8; 1/8; 1/8; 1/8; 1/8; 1/8];
end

% function [pl, ql, pr, qr] = OneDimBHbc(xl, ul, xr, ur, t)
%     global a5 theta1 thetaw;

%     % Robin boundary condition at x=0
%     pl = a5*(thetaw - ul); % Represents (h*theta - h*theta_w)
%     ql = ones(9, 1);

%     % Right boundary conditions (Dirichlet: u = 1)
%     pr = ur - theta1; % pr = ur - desired_value
%     qr = zeros(9, 1); % qr = 0 for Dirichlet
% end

function [pl,ql,pr,qr] = OneDimBHbc(xl,ul,xr,ur,t)
    global K om0 om1 om2 om3 om4 om5 om6 om7 a5 upsilon theta1 thetaw
    flusso = a5*(thetaw-ur(1));
    pl = [-flusso;
        -flusso-K*(ul(1)-ul(2));
        -flusso-K*(ul(1)-ul(3));
        -flusso-K*(ul(1)-ul(4));
        -flusso-K*(ul(1)-ul(5));
        -flusso-K*(ul(1)-ul(6));
        -flusso-K*(ul(1)-ul(7));
        -flusso-K*(ul(1)-ul(8));
        -flusso-K*(ul(1)-ul(9));
        0;0;0;0;0;0;0;0]; %flusso negativo, con osservatore - NOTA: CONTROLLA IL SEGNO!!!
    ql = [1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1];
    pr = [ur(1) - theta1; ur(2) - theta1; ur(3) - theta1; ur(4) - theta1; ur(5) - theta1; ur(6) - theta1; ur(7) - theta1; ur(8) - theta1;ur(9) - theta1;0;0;0;0;0;0;0;0];
    qr = [0;0;0;0;0;0;0;0;0;1;1;1;1;1;1;1;1];
    om0=upsilon*((ul(2)-ul(1)))^2;
    om1=upsilon*((ul(3)-ul(1)))^2;
    om2=upsilon*((ul(4)-ul(1)))^2;
    om3=upsilon*((ul(5)-ul(1)))^2;
    om4=upsilon*((ul(6)-ul(1)))^2;
    om5=upsilon*((ul(7)-ul(1)))^2;
    om6=upsilon*((ul(8)-ul(1)))^2;
    om7=upsilon*((ul(9)-ul(1)))^2;
end