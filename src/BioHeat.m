clear all
close all

global K lambda upsilon W W0 W1 W2 W3 W4 W5 W6 W7 b2 b3 theta10 theta20 theta30 theta_gt10 theta_gt20 X_gt1 X_gt2 om0 om1 om2  om3 om4  om5 om6 om7 a1 a2 a3 a4 a5

addpath('/Users/guglielmocappellini/Desktop/phd/code/readyaml')
% Default filename for YAML config
filename = 'config_matlab.yaml';

% Read and parse the YAML file into a MATLAB struct
config_data = readyaml(filename);
% Extract parameters from the struct
n_obs = config_data.model_parameters.n_obs;
L0 = config_data.model_properties.L0;
tauf = config_data.model_properties.tauf;
k = config_data.model_properties.k;
K = config_data.model_properties.K;
rho = config_data.model_properties.rho;
cp = config_data.model_properties.c;
t_room = config_data.model_properties.Troom;
t_y10 = config_data.model_properties.Ty10;
t_y20 = config_data.model_properties.Ty20;
t_y30 = config_data.model_properties.Ty30;
t_max = config_data.model_properties.Tmax;
t_gt10 = config_data.model_parameters.gt1_0;
t_gt20 = config_data.model_parameters.gt2_0;
h = config_data.model_properties.h;
b2 = config_data.model_properties.b2;
b3 = config_data.model_properties.b3;

% Observer weights based on the number of observers (3 or 8)
if n_obs == 3
    W0 = config_data.model_parameters.W0;
    W1 = config_data.model_parameters.W4;
    W2 = config_data.model_parameters.W7;
else 
    W0 = config_data.model_parameters.W0;
    W1 = config_data.model_parameters.W1;
    W2 = config_data.model_parameters.W2;
    W3 = config_data.model_parameters.W3;
    W4 = config_data.model_parameters.W4;
    W5 = config_data.model_parameters.W5;
    W6 = config_data.model_parameters.W6;
    W7 = config_data.model_parameters.W7;
end

% Other parameters
% delta = config_data.model_properties.delta;
lambda = config_data.model_parameters.lam;
upsilon = config_data.model_parameters.upsilon;

beta = config_data.model_properties.beta;
pwr_fact = config_data.model_properties.pwr_fact;
SAR_0 = config_data.model_properties.SAR_0;
PD = config_data.model_properties.PD;
x0 = config_data.model_properties.x0;

x_gt1 = config_data.model_parameters.x_gt1;
x_gt2 = config_data.model_parameters.x_gt2;

X_gt1 = x_gt1/L0;
X_gt2 = x_gt2/L0;

cc = log(2)/(PD - x0*10^(-2));
dT = t_max - t_room;

% Compute constants a1, a2, a3
a1 = (L0^2/tauf)*(rho*cp/k);
a2 = L0^2*cp/k;
a3 = pwr_fact*rho*(L0^2)*beta*SAR_0*exp(cc*x0)/k*dT;
a4 = cc*L0;
a5 = (h*L0)/k;
theta10 = (t_y10 - t_room)/(t_max - t_room);
theta20 = (t_y20 - t_room)/(t_max - t_room);
theta30 = (t_y30 - t_room)/(t_max - t_room);
theta_gt10 = (t_gt10 - t_room)/(t_max - t_room);
theta_gt20 = (t_gt20 - t_room)/(t_max - t_room);

% Initialize observer weights and constants
om0 = 0;
om1 = 0;
om2 = 0;
om3 = 0;
om4 = 0;
om5 = 0;
om6 = 0;
om7 = 0;

W = config_data.model_parameters.W4;

% Call the correct solver based on the number of observers
if n_obs == 3
    sol = OneDimBH_3Obs;  % Call the 3-observer case
elseif n_obs == 8
    sol = OneDimBH_8Obs;  % Call the default or other number of observers case
elseif n_obs == 1
    sol = OneDimBH_1Obs;  % Call the default or other number of observers case
end