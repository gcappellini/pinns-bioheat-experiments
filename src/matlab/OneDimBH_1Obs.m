function [sol] = OneDimBH_1Obs
    global output_path


    m = 0;
    x = linspace(0,1,101);
    t = linspace(0,1,101);
    
    sol = pdepe(m,@OneDimBHpde_1Obs,@OneDimBHic_1Obs,@OneDimBHbc_1Obs,x,t);

    u1 = sol(:,:,1); %soluzione del sistema
    u2 = sol(:,:,2); %soluzione dell'osservatore 0

    fileID = fopen(sprintf('%s/ground_truth/output_matlab_1Obs.txt', output_path),'w');
    
    for i = 1:101
       for j = 1:101
            
         fprintf(fileID,'%6.2f %6.2f %12.8f %12.8f\n', ...
         x(j), t(i), u1(i,j), u2(i,j));
    
            
       end
    end

    %-----------------
    function [c,f,s] = OneDimBHpde_1Obs(x,t,u,dudx)
    global a1 a2 a3 a4 W_obs W_sys
    %la prima equazione è quella del sistema, a seguire gli osservatori
    t
    c = [a1; a1];
    f = [1; 1].* dudx;
    
    % den=u(5)*exp(-om0)+u(6)*exp(-om1)+u(7)*exp(-om2);
    
    s = [-W_sys*a2*u(1)+a3*exp(-a4*x); 
        -W_obs*a2*u(2)+a3*exp(-a4*x);
        ];
    % --------------------------------------------------------------------------
    
        
    % --------------------------------------------------------------------------
    
    function u0 = OneDimBHic_1Obs(x)
    [theta0, thetahat0, ~, ~, ~] = ic_bc(x);
    
    u0 = [theta0; thetahat0];
    % --------------------------------------------------------------------------
    
    
    function [pl,ql,pr,qr] = OneDimBHbc_1Obs(xl,ul,xr,ur,t)
    global K a5
    
    % p(x,t,u) + q(x,t)f(x,t,u,dudx)=0

    [~, ~, y1, ~, ~] = ic_bc(xr);
    [~, ~, ~, ~, y3] = ic_bc(xl);
    flusso = a5*(y3 - ul(1));
    
    pl = [flusso; flusso-K*(ul(2)-ul(1))];
    ql = [1;1];

    pr = [ur(1) - y1; ur(2) - y1];
    
    qr = [0;0];
