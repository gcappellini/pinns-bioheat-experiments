function [sol] = OneDimBH

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

fileID = fopen('output_matlab_1.txt','w');

for i = 1:101
   for j = 1:101
        
     fprintf(fileID,'%6.2f %6.2f %12.8f %12.8f %12.8f %12.8f %12.8f\n', x(j), t(i), u1(i,j), u2(i,j), uav(i,j), u1(i,101));
        
   end
end

% surface plot of the system solution
figure;
surf(x,t,u1);
title('Numerical solution of the system.');
xlabel('Distance x');
ylabel('Time t');
saveas(gcf, '1_matlab_system.jpg');


% surface plot of the observer solution 
figure;
surf(x,t,uav);
title('Numerical solution of the observer.');
xlabel('Distance x');
ylabel('Time t');
saveas(gcf, '1_matlab_mm_obs.jpg');

figure
plot(t,u10,'r',t,u11,'g',t,u12,'b',t,u13,'cyan',t,u14,'.',t,u15,'-.',t,u16,'black',t,u17,'yellow') %plot the dynamic wheights
title('dynamic weights');
saveas(gcf, '1_matlab_weights.jpg');


% surface plot of the observation error 
figure;
surf(x,t,u1-uav);
title('Observation error with 100 mesh points.');
xlabel('Distance x');
ylabel('Time t');
saveas(gcf, '1_matlab_err_3d.jpg');



% Calculate the observation L2 error for each time step
l2_error_vs_time = sqrt(sum((u1 - uav).^2, 2));

% Plotting the 2D plot of L2 error vs. time
figure;
plot(t, l2_error_vs_time, 'LineWidth', 2);
title('L2 Observation Error vs. Time');
xlabel('Time t');
ylabel('L2 Error');
grid on;
saveas(gcf, '1_matlab_l2.jpg');


%solution profile at t_final
figure;
plot(x,u1(end,:),'o',x,u2(end,:),'r',x,u3(end,:),'g',x,u4(end,:),'b',x,u5(end,:),'cyan',...
    x,u6(end,:),'.',x,u7(end,:),'-.',x,u8(end,:),'black',x,u9(end,:),'yellow',x,uav(end,:),'x');

title('Solutions at t = t_{final}');
legend('System','Observer0','Observer1','Observer2','Observer3','Observer4','Observer5',...
    'Observer6','Observer7','ObserverMultiModel','Location', 'SouthWest');
xlabel('Distance x');
ylabel('temperature at t_{final}');
saveas(gcf, '1_matlab_tf.jpg');

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
saveas(gcf, '1_matlab_obs_err.jpg');

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
global lambda om0 om1 om2 om3 om4 om5 om6 om7 W W0 W1 W2 W3 W4 W5 W6 W7 a1 a2
%la prima equazione è quella del sistema, a seguire gli osservatori
t

c = [a1; a1; a1; a1; a1; a1; a1; a1; a1; 1; 1; 1; 1; 1; 1; 1; 1];
f = [1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1].* dudx;

den=u(10)*exp(-om0)+u(11)*exp(-om1)+u(12)*exp(-om2)+u(13)*exp(-om3)+...
    u(14)*exp(-om4)+u(15)*exp(-om5)+u(16)*exp(-om6)+u(17)*exp(-om7);

s = [-W*a2*u(1); 
    -W0*a2*u(2); 
    -W1*a2*u(3); 
    -W2*a2*u(4); 
    -W3*a2*u(5); 
    -W4*a2*u(6); 
    -W5*a2*u(7); 
    -W6*a2*u(8); 
    -W7*a2*u(9); 
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

function theta0 = sys_ic(x)
global a3 theta_w

%A = theta_w/(a3+1);
%bb = 1.0;
%theta0 = A*(1-x)*exp(-bb*x);
cc = 0.3;
bb = a3*(theta_w-cc);
aa = -(bb+cc)
theta0 = aa*x^2 + bb*x + cc;

% --------------------------------------------------------------------------

function u0 = OneDimBHic(x)
global K a3 delta theta_w

y1_0 = sys_ic(1);
y2_0 = sys_ic(0);
y3_0 = theta_w;
b1 = (a3*y3_0+(K-a3)*y2_0-(2+K)*delta)/(1+K);
ic_obs = y1_0 + b1*x + delta*x^2;
u0 = [sys_ic(x); ic_obs;  ic_obs; ic_obs; ic_obs; ic_obs; ic_obs; ic_obs; ic_obs; 1/8; 1/8; 1/8; 1/8; 1/8; 1/8; 1/8; 1/8];
% --------------------------------------------------------------------------


function [pl,ql,pr,qr] = OneDimBHbc(xl,ul,xr,ur,t)
global K om0 om1 om2 om3 om4 om5 om6 om7 upsilon a3 theta_w theta1
flusso = a3*(theta_w-ur(1));

pl = [-flusso;
    -flusso+K*(ul(1)-ul(2));
    -flusso+K*(ul(1)-ul(3));
    -flusso+K*(ul(1)-ul(4));
    -flusso+K*(ul(1)-ul(5));
    -flusso+K*(ul(1)-ul(6));
    -flusso+K*(ul(1)-ul(7));
    -flusso+K*(ul(1)-ul(8));
    -flusso+K*(ul(1)-ul(9));
    0;0;0;0;0;0;0;0];
ql = [1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1];
pr = [ur(1) - theta1; 
    ur(2) - theta1; 
    ur(3) - theta1; 
    ur(4) - theta1; 
    ur(5) - theta1; 
    ur(6) - theta1; 
    ur(7) - theta1; 
    ur(8) - theta1;
    ur(9) - theta1; 0;0;0;0;0;0;0;0];

qr = [0;0;0;0;0;0;0;0;0;1;1;1;1;1;1;1;1];
om0=upsilon*((ul(2)-ul(1)))^2;
om1=upsilon*((ul(3)-ul(1)))^2;
om2=upsilon*((ul(4)-ul(1)))^2;
om3=upsilon*((ul(5)-ul(1)))^2;
om4=upsilon*((ul(6)-ul(1)))^2;
om5=upsilon*((ul(7)-ul(1)))^2;
om6=upsilon*((ul(8)-ul(1)))^2;
om7=upsilon*((ul(9)-ul(1)))^2;


