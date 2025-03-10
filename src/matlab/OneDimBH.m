function [sol] = OneDimBH_1Obs
    global output_path


    m = 0;
    x = linspace(0,1,101);
    t = linspace(0,1,101);
    
    sol = pdepe(m,@OneDimBHpde_1Obs,@OneDimBHic_1Obs,@OneDimBHbc_1Obs,x,t);

    u1 = sol(:,:,1); %soluzione del sistema

    fileID = fopen(sprintf('%s.txt', output_path),'w');
    
    for i = 1:101
       for j = 1:101
            
         fprintf(fileID,'%6.2f %6.2f %12.8f\n', ...
         x(j), t(i), u1(i,j));
    
            
       end
    end

    %-----------------
    function [c,f,s] = OneDimBHpde_1Obs(x,t,u,dudx)
    global a1 a2 a3 a4 wbobs wbsys 
    %la prima equazione è quella del sistema, a seguire gli osservatori
    t
    % if x >= 0.2 && x <= 0.5
    % a1_ = a1 / incrfact;
    % a2_ = a2 / incrfact;
    % a3_ = a3 / incrfact;
    % else

    % end
    c = a1;
    f = dudx;
    
    % den=u(5)*exp(-om0)+u(6)*exp(-om1)+u(7)*exp(-om2);
    
    s = -wbsys*a2*u(1)+a3*exp(-a4*x);
    % --------------------------------------------------------------------------

    
    function u0 = OneDimBHic_1Obs(x)
    [theta0, ~, ~, ~, ~] = ic_bc(x, 0);
    
    u0 = theta0;
    % --------------------------------------------------------------------------
    
    
    function [pl,ql,pr,qr] = OneDimBHbc_1Obs(xl,ul,xr,ur,t)
    global oig a5
    
    % p(x,t,u) + q(x,t)f(x,t,u,dudx)=0

    [~, ~, y1, ~, ~] = ic_bc(xr, t);
    [~, ~, ~, ~, y3] = ic_bc(xl, t);
    flusso = a5*(y3 - ul(1));
    
    pl = flusso;
    ql = 1;

    pr = ur(1) - y1;
    
    qr = 0;
