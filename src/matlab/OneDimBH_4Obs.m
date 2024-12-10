function [sol] = OneDimBH_4Obs
    global output_path lambda upsilon

    m = 0;
    x = linspace(0,1,101);
    t = linspace(0,1,101);
    
    sol = pdepe(m,@OneDimBHpde_4Obs,@OneDimBHic_4Obs,@OneDimBHbc_4Obs,x,t);
    % % Extract the first solution component as u.  This is not necessary
    % % for a single equation, but makes a point about the form of the output.
    u1 = sol(:,:,1); %soluzione del sistema
    u2 = sol(:,:,2); %soluzione dell'osservatore 0
    u3 = sol(:,:,3); %soluzione dell'osservatore 1
    u4 = sol(:,:,4); %soluzione dell'osservatore 2
    u5 = sol(:,:,5); %soluzione dell'osservatore 3
    
    u10 = sol(:,:,6); %soluzione del peso 0
    u11 = sol(:,:,7); %soluzione del peso 1
    u12 = sol(:,:,8); %soluzione del peso 2
    u13 = sol(:,:,9); %soluzione del peso 3
    
    
    %multiple-model temperature estimation
    uav=u2.*u10+u3.*u11+u4.*u12 + u5.*u13;
    
    
    % Print Solution PDE
    filename = sprintf('%s/ground_truth/output_matlab_4Obs.txt', output_path);
    fileID = fopen(filename,'w');
    
    for i = 1:101
       for j = 1:101
            
         fprintf(fileID,'%6.2f %6.2f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f\n', ...
         x(j), t(i), u1(i,j), u2(i,j), u3(i,j), u4(i,j), u5(i,j), uav(i,j));
       end
    end
    
    
    filename2 = sprintf('%s/ground_truth/weights_l_%.3f_u_%.3f.txt', output_path, lambda, upsilon);
    fileID = fopen(filename2,'w');
    
    for i = 1:101
            
         fprintf(fileID,'%6.2f %12.8f %12.8f %12.8f %12.8f\n', ...
         t(i), u10(i,1), u11(i,1), u12(i,1), u13(i,1));
    
            
    end
    
    
    
    %-----------------
    function [c,f,s] = OneDimBHpde_4Obs(x,t,u,dudx)
    global lambda om0 om1 om2 om3 W_sys W0 W1 W2 W3 a1 a2 a3 a4
    %la prima equazione è quella del sistema, a seguire gli osservatoris
    t
    c = [a1; a1; a1; a1; a1; 1; 1; 1; 1];
    f = [1; 1; 1; 1; 1; 1; 1; 1; 1].* dudx;
    
    den=u(6)*exp(-om0)+u(7)*exp(-om1)+u(8)*exp(-om2)+u(9)*exp(-om3);

    s = [-W_sys*a2*u(1)+a3*exp(-a4*x); 
        -W0*a2*u(2)+a3*exp(-a4*x); 
        -W1*a2*u(3)+a3*exp(-a4*x); 
        -W2*a2*u(4)+a3*exp(-a4*x); 
        -W3*a2*u(5)+a3*exp(-a4*x); 
        -lambda*u(6)*(1-(exp(-om0)/den));
        -lambda*u(7)*(1-(exp(-om1)/den)); 
        -lambda*u(8)*(1-(exp(-om2)/den));
        -lambda*u(9)*(1-(exp(-om2)/den))
        ];
    % --------------------------------------------------------------------------

    
    function u0 = OneDimBHic_4Obs(x)
    [theta0, thetahat0, ~, ~, ~] = ic_bc(x);
    
    u0 = [theta0; thetahat0; thetahat0; thetahat0; thetahat0; 1/4; 1/4; 1/4; 1/4];
    % --------------------------------------------------------------------------
    
    
    function [pl,ql,pr,qr] = OneDimBHbc_4Obs(xl,ul,xr,ur,t)
    global K om0 om1 om2 om3 upsilon a5
    [~, ~, y1, ~, ~] = ic_bc(xr);
    [~, ~, ~, ~, y3] = ic_bc(xl);

    flusso = a5*(y3-ul(1));
    
    pl = [flusso;
        flusso-K*(ul(2)-ul(1));
        flusso-K*(ul(3)-ul(1));
        flusso-K*(ul(4)-ul(1));
        flusso-K*(ul(5)-ul(1));
        0;0;0;0];

    ql = [1;1;1;1;1;1;1;1;1];

    pr = [ur(1) - y1; 
        ur(2) - y1; 
        ur(3) - y1; 
        ur(4) - y1; 
        ur(5) - y1; 
        0;0;0; 0];
    
    qr = [0;0;0;0; 0; 1; 1;1;1];
    om0=upsilon*((ul(2)-ul(1)))^2;
    om1=upsilon*((ul(3)-ul(1)))^2;
    om2=upsilon*((ul(4)-ul(1)))^2;
    om3=upsilon*((ul(5)-ul(1)))^2;