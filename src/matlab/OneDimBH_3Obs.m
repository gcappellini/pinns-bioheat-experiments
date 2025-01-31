function [sol] = OneDimBH_3Obs
    global output_path lambda upsilon

    m = 0;
    x = linspace(0,1,101);
    t = linspace(0,1,101);
    
    sol = pdepe(m,@OneDimBHpde_3Obs,@OneDimBHic_3Obs,@OneDimBHbc_3Obs,x,t);
    % % Extract the first solution component as u.  This is not necessary
    % % for a single equation, but makes a point about the form of the output.
    u1 = sol(:,:,1); %soluzione del sistema
    u2 = sol(:,:,2); %soluzione dell'osservatore 0
    u3 = sol(:,:,3); %soluzione dell'osservatore 1
    u4 = sol(:,:,4); %soluzione dell'osservatore 2
    
    u10 = sol(:,:,5); %soluzione del peso 0
    u11 = sol(:,:,6); %soluzione del peso 1
    u12 = sol(:,:,7); %soluzione del peso 2
    
    
    %multiple-model temperature estimation
    uav=u2.*u10+u3.*u11+u4.*u12;
    
    
    % Print Solution PDE
    filename = sprintf('%s/output_matlab_3Obs.txt', output_path);
    fileID = fopen(filename,'w');
    
    for i = 1:101
       for j = 1:101
            
         fprintf(fileID,'%6.2f %6.2f %12.8f %12.8f %12.8f %12.8f %12.8f\n', ...
         x(j), t(i), u1(i,j), u2(i,j), u3(i,j), u4(i,j), uav(i,j));
       end
    end
    
    
    filename2 = sprintf('%s/weights_l_%.1f_u_%.1f_ground_truth.txt', output_path, lambda, upsilon);
    fileID = fopen(filename2,'w');
    
    for i = 1:101
            
         fprintf(fileID,'%6.2f %12.8f %12.8f %12.8f\n', ...
         t(i), u10(i,1), u11(i,1), u12(i,1));
    
            
    end
    
    
    
    %-----------------
    function [c,f,s] = OneDimBHpde_3Obs(x,t,u,dudx)
    global lambda om0 om1 om2 W_sys W0 W4 W7 a1 a2 a3 a4 incr_fact
    %la prima equazione è quella del sistema, a seguire gli osservatoris
    t
    if x >= 0.2 && x <= 0.5
        a1 = a1 / incr_fact;
        a2 = a2 / incr_fact;
        a3 = a3 / incr_fact;
    end
    c = [a1; a1; a1; a1; 1; 1; 1];
    f = [1; 1; 1; 1; 1; 1; 1].* dudx;
    
    den=u(5)*exp(-om0)+u(6)*exp(-om1)+u(7)*exp(-om2);
    
    s = [-W_sys*a2*u(1)+a3*exp(-a4*x); 
        -W0*a2*u(2)+a3*exp(-a4*x); 
        -W4*a2*u(3)+a3*exp(-a4*x); 
        -W7*a2*u(4)+a3*exp(-a4*x); 
        -lambda*u(5)*(1-(exp(-om0)/den));
        -lambda*u(6)*(1-(exp(-om1)/den)); 
        -lambda*u(7)*(1-(exp(-om2)/den))
        ];
    % --------------------------------------------------------------------------

    
    function u0 = OneDimBHic_3Obs(x)
    [theta0, thetahat0, ~, ~, ~] = ic_bc(x, 0);
    
    u0 = [theta0; thetahat0; thetahat0; thetahat0; 1/3; 1/3; 1/3];
    % --------------------------------------------------------------------------
    
    
    function [pl,ql,pr,qr] = OneDimBHbc_3Obs(xl,ul,xr,ur,t)
    global K om0 om1 om2 upsilon a5 str_exp
    [~, ~, y1, ~, ~] = ic_bc(xr, t);
    [~, ~, ~, y2, y3] = ic_bc(xl, t);

    if startsWith(str_exp, 'meas')
        sup_y = y2;
    else
        sup_y = ul(1);
    end

    flusso = a5*(y3-sup_y);
    
    pl = [flusso;
        flusso-K*(ul(2)-sup_y);
        flusso-K*(ul(3)-sup_y);
        flusso-K*(ul(4)-sup_y);
        0;0;0];
    ql = [1;1;1;1;1;1;1];
    pr = [ur(1) - y1; 
        ur(2) - y1; 
        ur(3) - y1; 
        ur(4) - y1; 
        0;0;0];
    
    qr = [0;0;0;0; 1;1;1];
    om0=upsilon*((ul(2)-sup_y))^2;
    om1=upsilon*((ul(3)-sup_y))^2;
    om2=upsilon*((ul(4)-sup_y))^2;