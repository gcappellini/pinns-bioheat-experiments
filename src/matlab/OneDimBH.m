function [sol] = OneDimBH
    global output_path

    m = 0;
    x = linspace(0,1,101);
    t = linspace(0,1,101);
    
    sol = pdepe(m,@OneDimBHpde,@OneDimBHic,@OneDimBHbc,x,t);

    u1 = sol(:,:,1); %soluzione del sistema

    fileID = fopen(sprintf('%s/output_matlab.txt', output_path),'w');
    
    for i = 1:101
       for j = 1:101
            
         fprintf(fileID,'%6.2f %6.2f %12.8f\n', ...
         x(j), t(i), u1(i,j));
    
            
       end
    end

    %-----------------
    function [c,f,s] = OneDimBHpde(x,t,u,dudx)
    global a1 a2 a3 a4 W_obs W_sys incr_fact
    %la prima equazione è quella del sistema, a seguire gli osservatori
    t
    f = dudx;
    % incr_fact=1.0;
    if x >= 0.2 && x <= 0.5
      a1 = a1 / incr_fact;
      a2 = a2 / incr_fact;
      a3 = a3 / incr_fact;
    end
    c = a1;

    s = -W_sys*a2*u(1)+a3*exp(-a4*x);
    % --------------------------------------------------------------------------

    
    function u0 = OneDimBHic(x)
    [theta0, ~, ~, ~, ~] = ic_bc(x, 0);
    
    u0 = theta0;
    % --------------------------------------------------------------------------
    
    
    function [pl,ql,pr,qr] = OneDimBHbc(xl,ul,xr,ur,t)
    global K a5
    
    % p(x,t,u) + q(x,t)f(x,t,u,dudx)=0

    [~, ~, y1, ~, ~] = ic_bc(xr, t);
    [~, ~, ~, ~, y3] = ic_bc(xl, t);
    flusso = a5*(y3 - ul(1));
    
    pl = flusso;
    ql = 1;
    pr = ur(1) - y1;
    qr = 0;
