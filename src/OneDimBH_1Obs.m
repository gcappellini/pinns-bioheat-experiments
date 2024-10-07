function [sol] = OneDimBH_1Obs

    m = 0;
    x = linspace(0,1,101);
    t = linspace(0,1,101);
    
    sol = pdepe(m,@OneDimBHpde_1Obs,@OneDimBHic_1Obs,@OneDimBHbc_1Obs,x,t);
    % % Extract the first solution component as u.  This is not necessary
    % % for a single equation, but makes a point about the form of the output.
    u1 = sol(:,:,1); %soluzione del sistema
    u2 = sol(:,:,2); %soluzione dell'osservatore 0
    % u3 = sol(:,:,3); %soluzione dell'osservatore 1
    % u4 = sol(:,:,4); %soluzione dell'osservatore 2
    % u5 = sol(:,:,5); %soluzione dell'osservatore 3
    % u6 = sol(:,:,6); %soluzione dell'osservatore 4
    % u7 = sol(:,:,7); %soluzione dell'osservatore 5
    % u8 = sol(:,:,8); %soluzione dell'osservatore 6
    % u9 = sol(:,:,9); %soluzione dell'osservatore 7
    
    % u10 = sol(:,:,5); %soluzione del peso 0
    % u11 = sol(:,:,6); %soluzione del peso 1
    % u12 = sol(:,:,7); %soluzione del peso 2
    % u13 = sol(:,:,13); %soluzione del peso 3
    % u14 = sol(:,:,14); %soluzione del peso 4
    % u15 = sol(:,:,15); %soluzione del peso 5
    % u16 = sol(:,:,16); %soluzione del peso 6
    % u17 = sol(:,:,17); %soluzione del peso 7
    
    
    %multiple-model temperature estimation
    % uav=u2.*u10+u3.*u11+u4.*u12;
    
    
    % Print Solution PDE
    
    fileID = fopen('output_matlab_1Obs.txt','w');
    
    for i = 1:101
       for j = 1:101
            
         fprintf(fileID,'%6.2f %6.2f %12.8f %12.8f\n', ...
         x(j), t(i), u1(i,j), u2(i,j));
    
            
       end
    end
    
    
    % fileID = fopen('weights_matlab_1Obs.txt','w');
    
    % for i = 1:101
            
    %      fprintf(fileID,'%6.2f %12.8f %12.8f %12.8f\n', ...
    %      t(i), u10(i,1), u11(i,1), u12(i,1));
    
            
    % end
    
    
    
    %-----------------
    function [c,f,s] = OneDimBHpde_1Obs(x,t,u,dudx)
    global lambda om0 om1 om2 W W0 W1 W2 a1 a2 a3 a4
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
    
    function thetahat0 = obs_ic(x)
        global a5 theta30 theta20 theta10 delta K
        b1 = delta;
        b4 = theta10;
        b3 = theta20 - b4;
        b2 = b3+K*(b3+b4)+a5*(theta30 - theta20)+ K*theta20;
        thetahat0 = (1-x)*(b1*x^2 + b2*x + b3) + b4;

    % function thetahat0 = obs_ic(x)
    % global a5 theta30 theta20 theta10 delta K
    % b2 = 3.9;
    % b3 = 5;
    % b4 = a5*(theta30-theta20);
    % b1 = (theta10-b4)*exp(b3);
    % thetahat0 = b1*x^(b2)*exp(-b3*x) + b4*x;

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
    global K om0 om1 om2 upsilon a5 theta30 theta10
    flusso = a5*(theta30-ul(1));
    
    pl = [flusso;
        flusso+K*(ul(1)-ul(2))
        ];
    ql = [1;1];
    pr = [ur(1) - theta10; 
        ur(2) - theta10
        ];
    
    qr = [0;0];
    % om0=upsilon*((ul(2)-ul(1)))^2;
    % om1=upsilon*((ul(3)-ul(1)))^2;
    % om2=upsilon*((ul(4)-ul(1)))^2;