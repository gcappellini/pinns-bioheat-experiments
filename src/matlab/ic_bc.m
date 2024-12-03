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


    
function thetahat0 = obs_ic(x)
    global b1 b2 b3 K

    thetahat0 = (b1 - x)*(b2 + b3 * exp(K*x));

function theta_y1 = theta_1()
    global theta10

    theta_y1=theta10;
    
function theta_y2 = theta_2()
    global theta20

    theta_y2=theta20;

function theta_y3 = theta_3()
    global theta30

    theta_y3 = theta30;

