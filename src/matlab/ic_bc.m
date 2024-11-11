function [theta0, thetahat0, theta_y1, theta_y2, theta_y3] = ic_bc(x)
    thetahat0 = obs_ic(x);
    theta0 = sys_ic(x);
    theta_y1 = theta_1();
    theta_y2 = theta_2();
    theta_y3 = theta_3();

function theta0 = sys_ic(x)
    global a5 theta30 theta20

    b = a5 * (theta30-theta20);
    c = theta20;
    a = -b -c;
    theta0 = a*x^2 + b*x + c;
    % theta0 = 0;

    
function thetahat0 = obs_ic(x)
    global b1 b2 K theta20 theta30 a5
    % b4 = a5 * (theta30 - theta20);
    % b1 = (theta10 - b4) * exp(b3);
    % thetahat0 = b1 .* (x.^b2) .* exp(-b3 .* x) + b4 .* x;
    % thetahat0 = (1-x^b1).*(exp(-50/(x+0.001))+b2);
    % thetahat0 = (1-x^b1).*(b2);
    A = a5 * (theta30 - theta20);
    B = theta20;
    C = b2 - (A - K.*B)/K;
    thetahat0 = (((A-K*B)/K)+C*exp(K*x))*(1-x)^(b1);

function theta_y1 = theta_1()
    global theta10

    % data = readmatrix(path_exp);    
    % tau = data(:, 1);
    % y1 = data(:, 2);
    % theta_y1 = interp1(tau, y1, t_vals, 'linear', 'extrap');
    theta_y1=theta10;
    
function theta_y2 = theta_2()
    % global path_exp

    % data = readmatrix(path_exp);
    % tau = data(:, 1);
    % y2 = data(:, 5);
    % theta_y2 = interp1(tau, y2, t_vals, 'spline', 'extrap');
    theta_y2 = 0.7;

function theta_y3 = theta_3()
    global theta30

    theta_y3 = theta30;

