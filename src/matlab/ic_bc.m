function [theta0, thetahat0, y1, y2, y3] = ic_bc(x, t)
    thetahat0 = obs_ic(x);
    theta0 = sys_ic(x);
    y1 = theta_1(t);
    y2 = theta_2();
    y3 = theta_3();

function theta0 = sys_ic(x)
    global a5 theta30 theta20 theta10  path_exp

    % b = a5 * (theta30-theta20);
    % c = theta20;
    % a = theta10 - b - c;
    % theta0 = a*x^2 + b*x + c;

    data = load(path_exp);
    first_row = data(1, :);
    temperatures = first_row(2:5);
    x_values = [1.0, 0.571, 0.143, 0.0];
    theta0 = interp1(x_values, temperatures, x, 'linear', 'extrap');

    
function first = g1(x)
    global a5 theta30 theta20 
    first = theta20 - a5 .* (theta30- theta20) .* x;

    
function second = g2(x, delta_x)
    global theta10
    second = (theta10 - g1(delta_x))*(x-delta_x)/(1-delta_x) + g1(delta_x);


function thetahat0 = obs_ic(x)
    global b1 b2 b3 K a5 theta30 theta20 theta10 delta_x

    thetahat0 = zeros(size(x)); % Initialize thetahat0 with the same size as x

    % Apply conditions element-wise
    % thetahat0(x <= delta_x) = g1(x(x <= delta_x));
    % thetahat0(x > delta_x) = g2(x(x > delta_x), delta_x);
    c = theta20;
    b = -a5 * (theta30 - theta20);
    a = theta10 - b - c;
    thetahat0 = a*x.^2 + b*x + c;



    
function y1 = theta_1(t)
    global theta10 path_exp
    data = load(path_exp);
    time = data(:, 1);
    measurement = data(:, 2);
    y1 = interp1(time, measurement, t, 'linear', 'extrap');

    
function y2 = theta_2()
    global theta20

    y2=theta20;

function y3 = theta_3()
    global theta30

    y3 = theta30;

