function [theta0, thetahat0, y1, y2, y3] = ic_bc(x, t)
    thetahat0 = obs_ic(x);
    theta0 = sys_ic(x);
    y1 = theta_1(t);
    y2 = theta_2();
    y3 = theta_3();

function theta0 = sys_ic(x)
    global a5 theta30 theta20 theta10

    b = a5 * (theta30-theta20);
    c = theta20;
    a = theta10 - b - c;
    theta0 = a*x^2 + b*x + c;


    
function thetahat0 = obs_ic(x)
    global b1 b2 b3 K

    thetahat0 = (b1 - x)*(b2 + b3 * exp(K*x));

    
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

