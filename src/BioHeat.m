clear all
close all

global K lambda delta upsilon W W0 W1 W2 W3 W4 W5 W6 W7 theta_w theta1 theta20 om0 om1 om2  om3 om4  om5 om6 om7 a1 a2 a3

% Default filename
filename1 = 'properties.json';
filename2 = 'parameters.json';

% Read the JSON file as a string
jsonData1 = fileread(filename1);
jsonData2 = fileread(filename2);

% Decode JSON data into a MATLAB struct
data = jsondecode(jsonData1);
data2 = jsondecode(jsonData2);

% Extract parameters from the struct
n_obs = data2.n_obs

L0 = data.L0;
tauf = data.tauf;
k = data.k;
K = data.K;
rho = data.rho;
cp = data.c;
t_room = data.Troom;
t_y20 = data.Ty20;
t_max = data.Tmax;
t_w = data.Twater;
h = data.h;

if n_obs == 3
    W0 = data2.W0;
    W1= data2.W4;
    W2 = data2.W7;
else 
    W0 = data2.W0;
    W1 = data2.W1;
    W2 = data2.W2;
    W3 = data2.W3;
    W4 = data2.W4;
    W5 = data2.W5;
    W6 = data2.W6;
    W7 = data2.W7;
end

delta = data.delta;
lambda = data2.lambda;
upsilon = data2.upsilon;

% Compute constants a1, a2, a3
a1 = (L0^2/tauf)*(rho*cp/k);
a2 = L0^2*cp/k;
a3 = (h*L0)/k;
theta_w = (t_w - t_room)/(t_max - t_room);
theta20 = (t_y20 - t_room)/(t_max - t_room);
theta1 = 0;

om0 = 0;
om1 = 0;
om2 = 0;
om3 = 0;
om4 = 0;
om5 = 0;
om6 = 0;
om7 = 0;

W = data2.W4;

if n_obs == 3
    sol = OneDimBH_3Obs;  % Call the 3-observer case
else
    sol = OneDimBH; % Call the default or other number of observers case
end