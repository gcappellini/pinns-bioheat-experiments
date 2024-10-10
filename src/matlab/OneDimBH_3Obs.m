function [sol] = OneDimBH_3Obs
    global str_exp

    src_dir = fileparts(cd);
    git_dir = fileparts(src_dir);
    output_path = sprintf('%s/tests/%s/ground_truth', git_dir, str_exp);

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
    
    
    filename2 = sprintf('%s/weights_matlab_3Obs.txt', output_path);
    fileID = fopen(filename2,'w');
    
    for i = 1:101
            
         fprintf(fileID,'%6.2f %12.8f %12.8f %12.8f\n', ...
         t(i), u10(i,1), u11(i,1), u12(i,1));
    
            
    end
    
    
    
    %-----------------
    function [c,f,s] = OneDimBHpde_3Obs(x,t,u,dudx)
    global lambda om0 om1 om2 W W0 W1 W2 a1 a2 a3 a4
    %la prima equazione è quella del sistema, a seguire gli osservatoris
    t
    c = [a1; a1; a1; a1; 1; 1; 1];
    f = [1; 1; 1; 1; 1; 1; 1].* dudx;
    
    den=u(5)*exp(-om0)+u(6)*exp(-om1)+u(7)*exp(-om2);
    
    s = [-W*a2*u(1)+a3*exp(-a4*x); 
        -W0*a2*u(2)+a3*exp(-a4*x); 
        -W1*a2*u(3)+a3*exp(-a4*x); 
        -W2*a2*u(4)+a3*exp(-a4*x); 
        -lambda*u(5)*(1-(exp(-om0)/den));
        -lambda*u(6)*(1-(exp(-om1)/den)); 
        -lambda*u(7)*(1-(exp(-om2)/den))
        ];
    % --------------------------------------------------------------------------

    
    function u0 = OneDimBHic_3Obs(x)
    [theta0, thetahat0] = ic_bc(x, 0);
    
    u0 = [theta0; thetahat0;  thetahat0; thetahat0; 1/3; 1/3; 1/3];
    % --------------------------------------------------------------------------
    
    
    function [pl,ql,pr,qr] = OneDimBHbc_3Obs(xl,ul,xr,ur,t)
    global K om0 om1 om2 upsilon a5
    [theta_y1, theta_y3] = ic_bc(0, t);
    flusso = a5*(theta_y3-ul(1));
    
    pl = [flusso;
        flusso+K*(ul(1)-ul(2));
        flusso+K*(ul(1)-ul(3));
        flusso+K*(ul(1)-ul(4));
        0;0;0];
    ql = [1;1;1;1;1;1;1];
    pr = [ur(1) - theta_y1; 
        ur(2) - theta_y1; 
        ur(3) - theta_y1; 
        ur(4) - theta_y1; 
        0;0;0];
    
    qr = [0;0;0;0; 1;1;1];
    om0=upsilon*((ul(2)-ul(1)))^2;
    om1=upsilon*((ul(3)-ul(1)))^2;
    om2=upsilon*((ul(4)-ul(1)))^2;