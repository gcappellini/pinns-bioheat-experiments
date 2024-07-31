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

fileID = fopen('output_matlab_ref.txt','w');

for i = 1:101
   for j = 1:101
        
     fprintf(fileID,'%6.2f %6.2f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f\n',...
         x(j), t(i), u1(i,j), Bolus(t(i)), u2(i,j), u3(i,j), u4(i,j),...
         u5(i,j), u6(i,j), u7(i,j), u8(i,j), u9(i,j), uav(i,j));
        
   end
end

%-----------------------------------
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
function u0 = OneDimBHic(x)
global K a4 a5

y1_0 = 0;
y2_0 = 0;
y3_0 = 0;
b1 = (a5*y3_0+(K-a5)*y2_0-(2+K)*a4)/(1+K);
ic_obs = y1_0 + b1*x + a4*x^2;
u0 = [0; ic_obs;  ic_obs; ic_obs; ic_obs; ic_obs; ic_obs; ic_obs; ic_obs; 1/8; 1/8; 1/8; 1/8; 1/8; 1/8; 1/8; 1/8];
% --------------------------------------------------------------------------

function ubol = Bolus(t)
c1 = 0.9;
c2 = 10.2;
c3 = 0.5;
c4 = 0.26;
ubol = (c3/(1+c1*exp(-c2*t)))-c4;

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


