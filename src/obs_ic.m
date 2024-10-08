function thetahat0 = obs_ic(x)
    global a5 theta30 theta20 theta10 b2 b3
    b4 = a5*(theta30-theta20);
    b1 = (theta10-b4)*exp(b3);
    thetahat0 = b1*x^(b2)*exp(-b3*x) + b4*x;
    
    % function thetahat0 = obs_ic(x)
    %     global theta10 theta_gt10 theta_gt20 theta20 X_gt1 X_gt2
        
    %     % Define the known points
    %     x_values = [1, X_gt1, X_gt2, 0];
    %     theta_values = [theta10, 0.99*theta_gt10, 0.99*theta_gt20, theta20];
        
    %     % Perform interpolation
    %     thetahat0 = interp1(x_values, theta_values, x, 'linear');
    