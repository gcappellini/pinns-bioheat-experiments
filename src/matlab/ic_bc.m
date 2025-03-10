function [theta0, thetahat0, y1, y2, y3] = ic_bc(x, t)
    global nobs
    if nobs == 0
        thetahat0 = 0 * x;
    else
        thetahat0 = obs_ic(x);
    end
    theta0 = sys_ic(x);
    y1 = theta_1(t);
    y2 = theta_2(t);
    y3 = theta_3();

function theta0 = sys_ic(x)
    global b1 b2 b3 b4 path_exp str_exp

    if startsWith(str_exp, 'meas')
        data = load(path_exp);
        first_row = data(1, :);
        temperatures = first_row(2:5);
        x_values = [1.0, 0.571, 0.143, 0.0];
        theta0 = interp1(x_values, temperatures, x, 'linear', 'extrap');

    else
        theta0 = b1.*x.^3 + b2.*x.^2 + b3.*x + b4;

    end



function thetahat0 = obs_ic(x)
    global c1 c2 c3

    thetahat0 = c1.*x.^2 + c2.*x + c3;

    
function y1 = theta_1(t)
    global y10 path_exp str_exp
    if startsWith(str_exp, 'meas')
        data = load(path_exp);
        time = data(:, 1);
        measurement_y1 = data(:, 2);
        y1 = interp1(time, measurement_y1, t, 'linear', 'extrap');
    else
        y1 = y10;
    end


    
function y2 = theta_2(t)
    global y20 path_exp str_exp
    if startsWith(str_exp, 'meas')
        data = load(path_exp);
        time = data(:, 1);
        measurement_y2 = data(:, 5);
        y2 = interp1(time, measurement_y2, t, 'linear', 'extrap');
    else
        y2 = y20;
    end

function y3 = theta_3()
    global y30

    y3 = y30;

