function [theta0, thetahat0, theta_y1, theta_y2, theta_y3] = ic_bc(x, t)
    thetahat0 = obs_ic(x);
    theta0 = sys_ic(x);
    theta_y1 = theta_1(t);
    theta_y2 = theta_2(t);
    theta_y3 = theta_3(t);

function thetahat0 = obs_ic(x)
    global a5 theta30 theta20 theta10 b2 b3

    b4 = a5 * (theta30 - theta20);
    b1 = (theta10 - b4) * exp(b3);

    thetahat0 = b1 .* (x.^b2) .* exp(-b3 .* x) + b4 .* x;

    
function theta0 = sys_ic(x)
    global theta10 theta_gt10 theta_gt20 theta20 X_gt1 X_gt2

    x_values = [1, X_gt1, X_gt2, 0];
    theta_values = [theta10, theta_gt10, theta_gt20, theta20];
    theta0 = interp1(x_values, theta_values, x, 'spline');

function theta_y1 = theta_1(t_vals)
    global path_exp

    data = readmatrix(path_exp);    
    tau = data(:, 1);
    y1 = data(:, 2);
    theta_y1 = interp1(tau, y1, t_vals, 'linear', 'extrap');
    
function theta_y2 = theta_2(t_vals)
    global path_exp

    data = readmatrix(path_exp);
    tau = data(:, 1);
    y2 = data(:, 5);
    theta_y2 = interp1(tau, y2, t_vals, 'spline', 'extrap');

function theta_y3 = theta_3(t_vals)

    theta_y3 = zeros(size(t_vals));

