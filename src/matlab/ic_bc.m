function [theta0, thetahat0, theta_y1, theta_y2, theta_y3] = ic_bc(x, t)
    thetahat0 = obs_ic(x);
    theta0 = sys_ic(x);
    theta_y1 = theta_1(t);
    theta_y2 = theta_2(t);
    theta_y3 = theta_3(t);

function theta0 = sys_ic(x)
    global a5 theta30 theta20

    b = a5 * (theta30-theta20);
    c = theta20;
    a = -b -c;
    theta0 = a*x^2 + b*x + c;
    % theta0 = 0;

    
function thetahat0 = obs_ic(x)
    global a5 theta30 theta20 theta10 b2 b3
    b4 = a5 * (theta30 - theta20);
    b1 = (theta10 - b4) * exp(b3);

    thetahat0 = b1 .* (x.^b2) .* exp(-b3 .* x) + b4 .* x;

function theta_y1 = theta_1(t_vals)
    % global path_exp

    % data = readmatrix(path_exp);    
    % tau = data(:, 1);
    % y1 = data(:, 2);
    % theta_y1 = interp1(tau, y1, t_vals, 'linear', 'extrap');
    theta_y1=zeros(size(t_vals));
    
function theta_y2 = theta_2(t_vals)
    % global path_exp

    % data = readmatrix(path_exp);
    % tau = data(:, 1);
    % y2 = data(:, 5);
    % theta_y2 = interp1(tau, y2, t_vals, 'spline', 'extrap');
    theta_y2 = 0.7*ones(size(t_vals));

function theta_y3 = theta_3(t_vals)

    theta_y3 = zeros(size(t_vals));

