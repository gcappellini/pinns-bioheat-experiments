function [sol] = OneDimBH_1Obs

    m = 0;
    x = linspace(0,1,101);
    t = linspace(0,1,101);
    
    sol = pdepe(m,@OneDimBHpde_1Obs,@OneDimBHic_1Obs,@OneDimBHbc_1Obs,x,t);

    u1 = sol(:,:,1); %soluzione del sistema
    u2 = sol(:,:,2); %soluzione dell'osservatore 0

    
    fileID = fopen('output_matlab_1Obs.txt','w');
    
    for i = 1:101
       for j = 1:101
            
         fprintf(fileID,'%6.2f %6.2f %12.8f %12.8f\n', ...
         x(j), t(i), u1(i,j), u2(i,j));
    
            
       end
    end

    
    
    %-----------------
    function [c,f,s] = OneDimBHpde_1Obs(x,t,u,dudx)
    global a1 a2 a3 a4
    %la prima equazione è quella del sistema, a seguire gli osservatoris
    t
    c = [a1; a1];
    f = [1; 1].* dudx;
    
    % den=u(5)*exp(-om0)+u(6)*exp(-om1)+u(7)*exp(-om2);
    
    s = [-W*a2*u(1)+a3*exp(-a4*x); 
        -W*a2*u(2)+a3*exp(-a4*x)
        ];
    % --------------------------------------------------------------------------
    
    function theta0 = sys_ic(x)
        global theta10 theta_gt10 theta_gt20 theta20 X_gt1 X_gt2
        
        % Define the known points
        x_values = [1, X_gt1, X_gt2, 0];
        theta_values = [theta10, theta_gt10, theta_gt20, theta20];
        
        % Perform interpolation
        theta0 = interp1(x_values, theta_values, x, 'linear');

    % function theta0 = sys_ic(x)
    %     global a5 theta30 theta20 theta10 delta K
    %     b1 = 0.9*delta;
    %     b4 = theta10;
    %     b3 = theta20 - b4;
    %     b2 = b3+K*(b3+b4)+a5*(theta30 - theta20)+ K*theta20;
    %     theta0 = (1-x)*(b1*x^2 + b2*x + b3) + b4;
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
        
    % --------------------------------------------------------------------------
    
    function u0 = OneDimBHic_1Obs(x)
    
    u0 = [sys_ic(x); obs_ic(x)];
    % --------------------------------------------------------------------------
    
    
    function [pl,ql,pr,qr] = OneDimBHbc_1Obs(xl,ul,xr,ur,t)
    global K a5 theta30 theta10
    flusso = a5*(theta30-ul(1));
    
    pl = [flusso;
        flusso+K*(ul(1)-ul(2))
        ];
    ql = [1;1];
    pr = [ur(1) - theta10; 
        ur(2) - theta10
        ];
    
    qr = [0;0];
