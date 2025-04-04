function [sol] = OneDimBH_8Obs
    global output_path ag ups

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
    filename = sprintf('%s.txt', output_path);
    fileID = fopen(filename,'w');
    
    for i = 1:101
       for j = 1:101
            
         fprintf(fileID,'%6.2f %6.2f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f\n', ...
         x(j), t(i), u1(i,j), u2(i,j), u3(i,j), u4(i,j), u5(i,j), u6(i,j), u7(i,j), u8(i,j), u9(i,j), uav(i,j));
    
            
       end
    end
    
    filename2 = sprintf('%s_weights.txt', output_path);
    fileID = fopen(filename2,'w');
    
    for i = 1:101
            
         fprintf(fileID,'%6.2f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f\n', ...
         t(i), u10(i,1), u11(i,1), u12(i,1), u13(i,1), u14(i,1), u15(i,1), u16(i,1), u17(i,1));
    
            
    end
    
    
    
    %-----------------
    function [c,f,s] = OneDimBHpde(x,t,u,dudx)
    global ag om0 om1 om2 om3 om4 om5 om6 om7 wb0 wb1 wb2 wb3 wb4 wb5 wb6 wb7 a1 a2 a3 a4 incr_fact
    %la prima equazione è quella del sistema, a seguire gli osservatori
    wb = perf(x,t);
    t
    %if x >= 0.2 && x <= 0.5
    %    a1 = a1 / incr_fact;
    %    a2 = a2 / incr_fact;
    %    a3 = a3 / incr_fact;
    %end
    c = [a1; a1; a1; a1; a1; a1; a1; a1; a1; 1; 1; 1; 1; 1; 1; 1; 1];
    f = [1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1].* dudx;
    
    den=u(10)*exp(-om0)+u(11)*exp(-om1)+u(12)*exp(-om2)+u(13)*exp(-om3)+...
        u(14)*exp(-om4)+u(15)*exp(-om5)+u(16)*exp(-om6)+u(17)*exp(-om7);
    
    s = [-wb*a2*u(1)+a3*exp(-a4*x); 
        -wb0*a2*u(2)+a3*exp(-a4*x); 
        -wb1*a2*u(3)+a3*exp(-a4*x); 
        -wb2*a2*u(4)+a3*exp(-a4*x); 
        -wb3*a2*u(5)+a3*exp(-a4*x); 
        -wb4*a2*u(6)+a3*exp(-a4*x); 
        -wb5*a2*u(7)+a3*exp(-a4*x); 
        -wb6*a2*u(8)+a3*exp(-a4*x); 
        -wb7*a2*u(9)+a3*exp(-a4*x); 
        -ag*u(10)*(1-(exp(-om0)/den));
        -ag*u(11)*(1-(exp(-om1)/den)); 
        -ag*u(12)*(1-(exp(-om2)/den)); 
        -ag*u(13)*(1-(exp(-om3)/den));
        -ag*u(14)*(1-(exp(-om4)/den));
        -ag*u(15)*(1-(exp(-om5)/den)); 
        -ag*u(16)*(1-(exp(-om6)/den)); 
        -ag*u(17)*(1-(exp(-om7)/den));
        ];
    % --------------------------------------------------------------------------
    
    function u0 = OneDimBHic(x)
    [theta0, thetahat0, ~, ~, ~] = ic_bc(x, 0);
    
    u0 = [theta0; thetahat0; thetahat0; thetahat0; thetahat0; thetahat0; thetahat0; thetahat0; thetahat0; 1/8; 1/8; 1/8; 1/8; 1/8; 1/8; 1/8; 1/8];
    % --------------------------------------------------------------------------
    
    
    function [pl,ql,pr,qr] = OneDimBHbc(xl,ul,xr,ur,t)
    global oig om0 om1 om2 om3 om4 om5 om6 om7 ups a5
    [~, ~, y1, ~, ~] = ic_bc(xr, t);
    [~, ~, ~, ~, y3] = ic_bc(xl, t);
    flusso = a5*(y3-ul(1));
    
    pl = [flusso;
        flusso+oig*(ul(1)-ul(2));
        flusso+oig*(ul(1)-ul(3));
        flusso+oig*(ul(1)-ul(4));
        flusso+oig*(ul(1)-ul(5));
        flusso+oig*(ul(1)-ul(6));
        flusso+oig*(ul(1)-ul(7));
        flusso+oig*(ul(1)-ul(8));
        flusso+oig*(ul(1)-ul(9));
        0;0;0;0;0;0;0;0];
    ql = [1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1];
    pr = [ur(1) - y1; 
        ur(2) - y1; 
        ur(3) - y1; 
        ur(4) - y1; 
        ur(5) - y1; 
        ur(6) - y1; 
        ur(7) - y1; 
        ur(8) - y1;
        ur(9) - y1; 0;0;0;0;0;0;0;0];
    
    qr = [0;0;0;0;0;0;0;0;0;1;1;1;1;1;1;1;1];
    om0=ups*((ul(2)-ul(1)))^2;
    om1=ups*((ul(3)-ul(1)))^2;
    om2=ups*((ul(4)-ul(1)))^2;
    om3=ups*((ul(5)-ul(1)))^2;
    om4=ups*((ul(6)-ul(1)))^2;
    om5=ups*((ul(7)-ul(1)))^2;
    om6=ups*((ul(8)-ul(1)))^2;
    om7=ups*((ul(9)-ul(1)))^2;