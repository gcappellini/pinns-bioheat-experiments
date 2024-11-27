function [sol] = OneDimBH_8Obs
    global output_path lambda upsilon

    m = 0;
    x = linspace(0,1,101);
    t = linspace(0,1,101);
    
    sol = pdepe(m,@OneDimBHpde,@OneDimBHic,@OneDimBHbc,x,t);
    % % Extract the first solution component as u.  This is not necessary
    % % for a single equation, but makes a point about the form of the output.
    u1 = sol(:,:,1); %soluzione del sistema
    u2 = sol(:,:,2); %soluzione dell'osservatore 0
    u3 = sol(:,:,3); %soluzione dell'osservatore 1
    u4 = sol(:,:,4); %soluzione dell'osservatore 2
    u5 = sol(:,:,5); %soluzione dell'osservatore 3
    u6 = sol(:,:,6); %soluzione dell'osservatore 4
    u7 = sol(:,:,7); %soluzione dell'osservatore 5
    u8 = sol(:,:,8); %soluzione dell'osservatore 6
    u9 = sol(:,:,9); %soluzione dell'osservatore 7
    
    u10 = sol(:,:,10); %soluzione del peso 0
    u11 = sol(:,:,11); %soluzione del peso 1
    u12 = sol(:,:,12); %soluzione del peso 2
    u13 = sol(:,:,13); %soluzione del peso 3
    u14 = sol(:,:,14); %soluzione del peso 4
    u15 = sol(:,:,15); %soluzione del peso 5
    u16 = sol(:,:,16); %soluzione del peso 6
    u17 = sol(:,:,17); %soluzione del peso 7
    
    
    %multiple-model temperature estimation
    uav=u2.*u10+u3.*u11+u4.*u12+u5.*u13+u6.*u14+u7.*u15+u8.*u16+u9.*u17;
    
    
    % Print Solution PDE
    filename = sprintf('%s/ground_truth/output_matlab_8Obs.txt', output_path);
    fileID = fopen(filename,'w');
    
    for i = 1:101
       for j = 1:101
            
         fprintf(fileID,'%6.2f %6.2f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f\n', ...
         x(j), t(i), u1(i,j), u2(i,j), u3(i,j), u4(i,j), u5(i,j), u6(i,j), u7(i,j), u8(i,j), u9(i,j), uav(i,j));
    
            
       end
    end
    
    filename2 = sprintf('%s/ground_truth/weights_l_%d_u_%d.txt', output_path, lambda, upsilon);
    fileID = fopen(filename2,'w');
    
    for i = 1:101
            
         fprintf(fileID,'%6.2f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f\n', ...
         t(i), u10(i,1), u11(i,1), u12(i,1), u13(i,1), u14(i,1), u15(i,1), u16(i,1), u17(i,1));
    
            
    end
    
    
    
    %-----------------
    function [c,f,s] = OneDimBHpde(x,t,u,dudx)
    global lambda om0 om1 om2 om3 om4 om5 om6 om7 W_sys W0 W1 W2 W3 W4 W5 W6 W7 a1 a2 a3 a4
    %la prima equazione è quella del sistema, a seguire gli osservatoris
    t
    c = [a1; a1; a1; a1; a1; a1; a1; a1; a1; 1; 1; 1; 1; 1; 1; 1; 1];
    f = [1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1].* dudx;
    
    den=u(10)*exp(-om0)+u(11)*exp(-om1)+u(12)*exp(-om2)+u(13)*exp(-om3)+...
        u(14)*exp(-om4)+u(15)*exp(-om5)+u(16)*exp(-om6)+u(17)*exp(-om7);
    
    s = [-W_sys*a2*u(1)+a3*exp(-a4*x); 
        -W0*a2*u(2)+a3*exp(-a4*x); 
        -W1*a2*u(3)+a3*exp(-a4*x); 
        -W2*a2*u(4)+a3*exp(-a4*x); 
        -W3*a2*u(5)+a3*exp(-a4*x); 
        -W4*a2*u(6)+a3*exp(-a4*x); 
        -W5*a2*u(7)+a3*exp(-a4*x); 
        -W6*a2*u(8)+a3*exp(-a4*x); 
        -W7*a2*u(9)+a3*exp(-a4*x); 
        -lambda*u(10)*(1-(exp(-om0)/den));
        -lambda*u(11)*(1-(exp(-om1)/den)); 
        -lambda*u(12)*(1-(exp(-om2)/den)); 
        -lambda*u(13)*(1-(exp(-om3)/den));
        -lambda*u(14)*(1-(exp(-om4)/den));
        -lambda*u(15)*(1-(exp(-om5)/den)); 
        -lambda*u(16)*(1-(exp(-om6)/den)); 
        -lambda*u(17)*(1-(exp(-om7)/den));
        ];
    % --------------------------------------------------------------------------
    
    function u0 = OneDimBHic(x)
    [theta0, thetahat0, ~, ~, ~] = ic_bc(x);
    
    u0 = [theta0; thetahat0; thetahat0; thetahat0; thetahat0; thetahat0; thetahat0; thetahat0; thetahat0; 1/8; 1/8; 1/8; 1/8; 1/8; 1/8; 1/8; 1/8];
    % --------------------------------------------------------------------------
    
    
    function [pl,ql,pr,qr] = OneDimBHbc(xl,ul,xr,ur,t)
    global K om0 om1 om2 om3 om4 om5 om6 om7 upsilon a5
    [~, ~, theta_y1, ~, ~] = ic_bc(xr);
    [~, ~, ~, ~, theta_y3] = ic_bc(xl);
    flusso = a5*(theta_y3-ul(1));
    
    pl = [flusso;
        flusso+K*(ul(1)-ul(2));
        flusso+K*(ul(1)-ul(3));
        flusso+K*(ul(1)-ul(4));
        flusso+K*(ul(1)-ul(5));
        flusso+K*(ul(1)-ul(6));
        flusso+K*(ul(1)-ul(7));
        flusso+K*(ul(1)-ul(8));
        flusso+K*(ul(1)-ul(9));
        0;0;0;0;0;0;0;0];
    ql = [1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1];
    pr = [ur(1) - theta_y1; 
        ur(2) - theta_y1; 
        ur(3) - theta_y1; 
        ur(4) - theta_y1; 
        ur(5) - theta_y1; 
        ur(6) - theta_y1; 
        ur(7) - theta_y1; 
        ur(8) - theta_y1;
        ur(9) - theta_y1; 0;0;0;0;0;0;0;0];
    
    qr = [0;0;0;0;0;0;0;0;0;1;1;1;1;1;1;1;1];
    om0=upsilon*((ul(2)-ul(1)))^2;
    om1=upsilon*((ul(3)-ul(1)))^2;
    om2=upsilon*((ul(4)-ul(1)))^2;
    om3=upsilon*((ul(5)-ul(1)))^2;
    om4=upsilon*((ul(6)-ul(1)))^2;
    om5=upsilon*((ul(7)-ul(1)))^2;
    om6=upsilon*((ul(8)-ul(1)))^2;
    om7=upsilon*((ul(9)-ul(1)))^2;