clear all
close all

global K lambda delta upsilon W W0 W1 W2 W3 W4 W5 W6 W7 theta_w theta1 theta20 om0 om1 om2  om3 om4  om5 om6 om7 a1 a2 a3

addpath('/Users/guglielmocappellini/Desktop/phd/code/yamlmatlab-master')
% Default filename for YAML config
filename = 'config.yaml';

% Read and parse the YAML file into a MATLAB struct
config_data = ReadYaml(filename);
% Extract parameters from the struct
n_obs = config_data.model_parameters.n_obs;
L0 = config_data.model_properties.L0;
tauf = config_data.model_properties.tauf;
k = config_data.model_properties.k;
K = config_data.model_properties.K;
rho = config_data.model_properties.rho;
cp = config_data.model_properties.c;
t_room = config_data.model_properties.Troom;
t_y20 = config_data.model_properties.Ty20;
t_max = config_data.model_properties.Tmax;
t_w = config_data.model_properties.Twater;
h = config_data.model_properties.h;

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
delta = config_data.model_properties.delta;
lambda = config_data.model_parameters.lam;
upsilon = config_data.model_parameters.upsilon;

% Compute constants a1, a2, a3
a1 = (L0^2/tauf)*(rho*cp/k);
a2 = L0^2*cp/k;
a3 = (h*L0)/k;
theta_w = (t_w - t_room)/(t_max - t_room);
theta20 = (t_y20 - t_room)/(t_max - t_room);
theta1 = 0;

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
else
    sol = OneDimBH;  % Call the default or other number of observers case
end