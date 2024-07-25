
function [sol] = OneDimBH
global tf

m = 0;
x = linspace(0,1,101);
t = linspace(0,tf,101);

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

fileID = fopen('output_matlab.txt','w');

for i = 1:101
   for j = 1:101
        
     fprintf(fileID,'%6.2f %6.2f %12.8f %12.8f %12.8f %12.8f %12.8f\n', x(j), t(i), u1(i,j), u2(i,j), uav(i,j), u1(i,101), Bolus(t(i)));
        
   end
end

% surface plot of the system solution
figure;
surf(x,t,u1);
title('Numerical solution of the system.');
xlabel('Distance x');
ylabel('Time t');


% surface plot of the observer solution 
figure;
surf(x,t,uav);
title('Numerical solution of the observer.');
xlabel('Distance x');
ylabel('Time t');

figure
plot(t,u10,'r',t,u11,'g',t,u12,'b',t,u13,'cyan',t,u14,'.',t,u15,'-.',t,u16,'black',t,u17,'yellow') %plot the dynamic wheights
title('dynamic weights');


% surface plot of the observation error 
figure;
surf(x,t,u1-uav);
title('Observation error with 100 mesh points.');
xlabel('Distance x');
ylabel('Time t');

% Calculate the observation error
error = abs(u1 - uav);

% Sum the errors over the spatial dimension (x-axis) for each time step
sum_error_vs_time = sum(error, 2);

% Plotting the 2D plot of summed error vs. time
figure;
plot(t, sum_error_vs_time, 'LineWidth', 2);
title('Summed Observation Error vs. Time');
xlabel('Time t');
ylabel('Summed Error');
grid on;

% Optionally, add more plot settings
legend('Summed Error', 'Location', 'best');


%solution profile at t_final
figure;
plot(x,u1(end,:),'o',x,u2(end,:),'r',x,u3(end,:),'g',x,u4(end,:),'b',x,u5(end,:),'cyan',...
    x,u6(end,:),'.',x,u7(end,:),'-.',x,u8(end,:),'black',x,u9(end,:),'yellow',x,uav(end,:),'x');

title('Solutions at t = t_{final}');
legend('System','Observer0','Observer1','Observer2','Observer3','Observer4','Observer5',...
    'Observer6','Observer7','ObserverMultiModel','Location', 'SouthWest');
xlabel('Distance x');
ylabel('temperature at t_{final}');

omegas = calculateOmegas(sol, x, t);

% Plot omega evolution
figure;
plot(t, omegas(:, 1), 'r-', 'LineWidth', 1.5, 'DisplayName', 'e_0'); hold on;
plot(t, omegas(:, 2), 'g-', 'LineWidth', 1.5, 'DisplayName', 'e_1');
plot(t, omegas(:, 3), 'b-', 'LineWidth', 1.5, 'DisplayName', 'e_2');
plot(t, omegas(:, 4), 'c-', 'LineWidth', 1.5, 'DisplayName', 'e_3');
plot(t, omegas(:, 5), 'm-', 'LineWidth', 1.5, 'DisplayName', 'e_4');
plot(t, omegas(:, 6), '-.', 'LineWidth', 1.5, 'DisplayName', 'e_5');
plot(t, omegas(:, 7), 'k-', 'LineWidth', 1.5, 'DisplayName', 'e_6');
plot(t, omegas(:, 8), 'y-', 'LineWidth', 1.5, 'DisplayName', 'e_7');
hold off;
title('Evolution of Observation errors');
xlabel('Time t');
ylabel('\omega Values');
legend('show', 'Location', 'best');
grid on;

% Helper function to calculate omegas
function omegas = calculateOmegas(sol, x, t)
global upsilon
    numSteps = length(t);
    omegas = zeros(numSteps, 8); % Preallocate matrix for omega values

    for i = 1:numSteps
        ur = sol(i, end, :); % Right boundary values at each time step
        omegas(i, 1) = upsilon * (ur(2) - ur(1))^2;
        omegas(i, 2) = upsilon * (ur(3) - ur(1))^2;
        omegas(i, 3) = upsilon * (ur(4) - ur(1))^2;
        omegas(i, 4) = upsilon * (ur(5) - ur(1))^2;
        omegas(i, 5) = upsilon * (ur(6) - ur(1))^2;
        omegas(i, 6) = upsilon * (ur(7) - ur(1))^2;
        omegas(i, 7) = upsilon * (ur(8) - ur(1))^2;
        omegas(i, 8) = upsilon * (ur(9) - ur(1))^2;
    end


%-----------------
function [c,f,s] = OneDimBHpde(x,t,u,dudx)
global lambda om0 om1 om2 om3 om4 om5 om6 om7 W W0 W1 W2 W3 W4 W5 W6 W7 a1 a2 a3 a6
%la prima equazione è quella del sistema, a seguire gli osservatori
t

c = [a1; a1; a1; a1; a1; a1; a1; a1; a1; 1; 1; 1; 1; 1; 1; 1; 1];
f = [1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1].* dudx;

den=u(10)*exp(-om0)+u(11)*exp(-om1)+u(12)*exp(-om2)+u(13)*exp(-om3)+...
    u(14)*exp(-om4)+u(15)*exp(-om5)+u(16)*exp(-om6)+u(17)*exp(-om7);

s = [-W*a2*u(1)+a3*exp(-a6*(1-x)); 
    -W0*a2*u(2)+a3*exp(-a6*(1-x)); 
    -W1*a2*u(3)+a3*exp(-a6*(1-x)); 
    -W2*a2*u(4)+a3*exp(-a6*(1-x)); 
    -W3*a2*u(5)+a3*exp(-a6*(1-x)); 
    -W4*a2*u(6)+a3*exp(-a6*(1-x)); 
    -W5*a2*u(7)+a3*exp(-a6*(1-x)); 
    -W6*a2*u(8)+a3*exp(-a6*(1-x)); 
    -W7*a2*u(9)+a3*exp(-a6*(1-x)); 
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
function icsys = Initial(x)
e1 = -2.6;
e2 = 0.9;
e3 = 0.6;
icsys = e1*(x-e3)^2+e2;

function u0 = OneDimBHic(x)
global K a4 a5

y1_0 = Initial(0);
y2_0 = Initial(1);
y3_0 = 0.438;
b1 = (a5*y3_0+(K-a5)*y2_0-(2+K)*a4)/(1+K);
ic_obs = y1_0 + b1*x + a4*x^2;
u0 = [Initial(x); ic_obs;  ic_obs; ic_obs; ic_obs; ic_obs; ic_obs; ic_obs; ic_obs; 1/8; 1/8; 1/8; 1/8; 1/8; 1/8; 1/8; 1/8];
% --------------------------------------------------------------------------

function ubol = Bolus(t)
c1 = 0.9;
c2 = 10.2;
c3 = 0.5;
c4 = 0.26;
%ubol = (c3/(1+c1*exp(-c2*t)))-c4;
ubol=0.438;

% --------------------------------------------------------------------------

function [pl,ql,pr,qr] = OneDimBHbc(xl,ul,xr,ur,t)
global K om0 om1 om2 om3 om4 om5 om6 om7 a5 upsilon
flusso = a5*(Bolus(t)-ur(1));
pl = [ul(1);ul(2);ul(3);ul(4);ul(5);ul(6);ul(7);ul(8);ul(9);0;0;0;0;0;0;0;0];
ql = [0;0;0;0;0;0;0;0;0;1;1;1;1;1;1;1;1];
pr = [-flusso;
    -flusso-K*(ur(1)-ur(2));
    -flusso-K*(ur(1)-ur(3));
    -flusso-K*(ur(1)-ur(4));
    -flusso-K*(ur(1)-ur(5));
    -flusso-K*(ur(1)-ur(6));
    -flusso-K*(ur(1)-ur(7));
    -flusso-K*(ur(1)-ur(8));
    -flusso-K*(ur(1)-ur(9));
    0;0;0;0;0;0;0;0]; %flusso negativo, con osservatore
qr = [1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1];
om0=upsilon*((ur(2)-ur(1)))^2;
om1=upsilon*((ur(3)-ur(1)))^2;
om2=upsilon*((ur(4)-ur(1)))^2;
om3=upsilon*((ur(5)-ur(1)))^2;
om4=upsilon*((ur(6)-ur(1)))^2;
om5=upsilon*((ur(7)-ur(1)))^2;
om6=upsilon*((ur(8)-ur(1)))^2;
om7=upsilon*((ur(9)-ur(1)))^2;


