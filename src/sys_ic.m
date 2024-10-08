function theta0 = sys_ic(x)
    global theta10 theta_gt10 theta_gt20 theta20 X_gt1 X_gt2
    
    % Define the known points
    x_values = [1, X_gt1, X_gt2, 0];
    theta_values = [theta10, theta_gt10, theta_gt20, theta20];
    
    % Perform interpolation
    theta0 = interp1(x_values, theta_values, x, 'linear');

% --------------------------------------------------------------------------
% function thetahat0 = obs_ic(x)
%     global a5 theta30 theta20 theta10
%     x_ch = 0.5
%     % Define the piecewise function
%     if x >= 0 && x <= x_ch
%         % Line from 0 to x_ch with steepness a5*(theta30 - theta20)
%         thetahat0 = theta20 + a5 * (theta30 - theta20) * (x / x_ch);
%     else
%         slope = (theta10 - (theta20 + a5 * (theta30 - theta20))) / (1 - x_ch);
%         thetahat0 = theta20 + a5 * (theta30 - theta20) + slope * (x - x_ch);
%     end

% function thetahat0 = obs_ic(x)
%     global a5 theta30 theta20 theta10 delta K
%     b1 = delta;
%     b4 = theta10;
%     b3 = theta20 - b4;
%     b2 = b3+K*(b3+b4)+a5*(theta30 - theta20)+ K*theta20;
%     thetahat0 = (1-x)*(b1*x^2 + b2*x + b3) + b4;

