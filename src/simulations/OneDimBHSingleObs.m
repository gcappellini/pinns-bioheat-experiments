%From https://es.mathworks.com/help/matlab/math/solve-single-pde.html

function [sol] = OneDimBHSingleObs
global k normerr a1 a2 a3 a4 a5 v


k = 4;

m = 0;
a1 = 0.942174066658815;
a2 = 4.1722456718878815;
a3 = 0;
a4= 0.5;
a5= 1.875;
a6 = 2.0;

v = @(t) a6;
x = 0:0.01:1; % 100 valori tra 0 e 1
t = 0:0.01:1; % 100 valori tra 0 e 1

sol = pdepe(m,@OneDimBHpde,@OneDimBHic,@OneDimBHbc,x,t);
% % Extract the first solution component as u.  This is not necessary
% % for a single equation, but makes a point about the form of the output.
u1 = sol(:,:,1); %soluzione del sistema
u2 = sol(:,:,2); %soluzione dell'osservatore 1

% Print Solution PDE

fileID = fopen('file0-cdc.txt','w');
%fprintf(fileID,'%6s %12s\n','x','exp(x)');
%fprintf(fileID,'%6.2f %12.8f %12.8f\n', t, x, u);

for i = 1:101
   for j = 1:101
        
     fprintf(fileID,'%6.2f %6.2f %6.2f %12.8f\n', x(j), t(i), v(t(i)), u1(i,j));
        
   end
end

% Print solution observer

fileID = fopen('output_matlab_observer.txt','w');
%fprintf(fileID,'%6s %12s\n','x','exp(x)');
%fprintf(fileID,'%6.2f %12.8f %12.8f\n', t, x, u);

for i = 1:101
   for j = 1:101
        
     fprintf(fileID,'%6.2f %6.2f %12.8f\n', x(j), t(i), u2(i,j));
        
   end
end



% surface plot of the system solution
figure;
surf(x,t,u1);
title('Numerical solution of the system computed with 100 mesh points.');
xlabel('Distance x');
ylabel('Time t');
% surface plot of the observer solution 
figure;
surf(x,t,u2);
title('Numerical solution of the observer computed with 100 mesh points.');
xlabel('Distance x');
ylabel('Time t');

% surface plot of the observer solution 
figure;
surf(x,t,u1-u2);
title('Observation error with 100 mesh points.');
xlabel('Distance x');
ylabel('Time t');

%solution profile at t_t_final
figure;
plot(x,u1(end,:),'o',x,u2(end,:),'x');

title('Solutions at t = t_{final}.');
legend('System','Observer1','Location', 'SouthWest');
xlabel('Distance x');
ylabel('temperature at t_{final}');
err=u1-u2;
for i=1:100
   normerr=[normerr;
   norm(err(i,:),2)];
end



%-----------------

% Code equation

function [c,f,s] = OneDimBHpde(x,t,u,dudx)
global a1 a2
%La prima equazione è quella del sistema, a seguire gli osservatori

c = [1; 1];
f = [a1; a1].* dudx;
s = [-u(1)*a2; 
     -u(2)*a2; 
    ];

% --------------------------------------------------------------------------

% Code initial conditions

function u0 = OneDimBHic(x)
global a4 a5

u0 = [a4*x^4+a5*((x-1)^2)*x; a4*x^4];


% --------------------------------------------------------------------------

% Code boundary conditions

function [pl,ql,pr,qr] = OneDimBHbc(xl,ul,xr,ur,t)
global k v
pl = [ul(1); ul(2)];
ql = [0; 0];
pr = [-v(t); -v(t)-k*(ur(1)-ur(2))];
qr = [1;1];
