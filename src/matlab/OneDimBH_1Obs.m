function [sol] = OneDimBH_1Obs
    global output_path


    m = 0;
    x = linspace(0,1,101);
    t = linspace(0,1,101);
    
    sol = pdepe(m,@OneDimBHpde_1Obs,@OneDimBHic_1Obs,@OneDimBHbc_1Obs,x,t);

    u1 = sol(:,:,1); %soluzione del sistema
    u2 = sol(:,:,2); %soluzione dell'osservatore 0

    
    filename = sprintf('%s/output_matlab_1Obs.txt', output_path);
    fileID = fopen(filename,'w');
    
    for i = 1:101
       for j = 1:101
            
         fprintf(fileID,'%6.2f %6.2f %12.8f %12.8f\n', ...
         x(j), t(i), u1(i,j), u2(i,j));
    
            
       end
    end

    %-----------------
    function [c,f,s] = OneDimBHpde_1Obs(x,t,u,dudx)
    global a1 a2 a3 a4 W
    %la prima equazione è quella del sistema, a seguire gli osservatoris
    t
    c = [a1; a1];
    f = [1; 1].* dudx;
    
    % den=u(5)*exp(-om0)+u(6)*exp(-om1)+u(7)*exp(-om2);
    
    s = [-W*a2*u(1)+a3*exp(-a4*x); 
        -W*a2*u(2)+a3*exp(-a4*x);
        ];
    % --------------------------------------------------------------------------
    
        
    % --------------------------------------------------------------------------
    
    function u0 = OneDimBHic_1Obs(x)
    [theta0, thetahat0] = ic_bc(x, 0);
    
    u0 = [theta0; thetahat0];
    % --------------------------------------------------------------------------
    
    
    function [pl,ql,pr,qr] = OneDimBHbc_1Obs(xl,ul,xr,ur,t)
    global K a5 theta30 theta10
    % [theta_y1, theta_y3] = ic_bc(0, t);
    flusso = a5*(theta30-ul(1));
    
    pl = [flusso;
        flusso+K*(ul(1)-ul(2));
        ];
    ql = [1;1];
    pr = [ur(1) - theta10; 
        ur(2) - theta10;
        ];
    
    qr = [0;0];
