function theta_1 = theta_1(x)
    global str_exp

    % Load the text file
    data = readmatrix(str_exp);

    % Extract the first two columns
    tau = data(:, 1);
    y1 = data(:, 2);
    x = linspace(0,1,101);
    
    % Perform interpolation
    theta_1 = interp1(tau, y1, x, 'linear');



