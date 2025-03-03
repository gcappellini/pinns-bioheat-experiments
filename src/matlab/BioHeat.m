clear all
close all

global oig lambda upsilon wbobs wbsys wb0 wb1 wb2 wb3 wb4 wb5 wb6 wb7 c1 c2 c3 b1 b2 b3 b4 theta10 theta20 theta30 incrfact thetagt10 thetagt0 Xgt1 Xgt om0 om1 om2 om3 om4 om5 om6 om7 a1 a2 a3 a4 a5 str_exp output_path nobs

% Replace the following with the path to readyaml (find link on the internet)
addpath('/Users/guglielmocappellini/Desktop/research/code/readyaml')

src_dir = fileparts(cd);
git_dir = fileparts(src_dir);
filename = sprintf('%s/src/configs/config_ground_truth.yaml', git_dir);

% Read and parse the YAML file into a MATLAB struct
config_data = readyaml(filename);
% Extract parameters from the struct
nobs = config_data.parameters.nobs;
theta10 = config_data.pdecoeff.theta10;
thetagt0 = config_data.pdecoeff.thetagt0;
theta20 = config_data.pdecoeff.theta20;
theta30 = config_data.pdecoeff.theta30;
thetagt10 = config_data.pdecoeff.thetagt10;
a1 = config_data.pdecoeff.a1;
a2 = config_data.pdecoeff.a2;
a3 = config_data.pdecoeff.a3;
a4 = config_data.pdecoeff.a4;
a5 = config_data.pdecoeff.a5;
b1 = config_data.pdecoeff.b1;
b2 = config_data.pdecoeff.b2;
b3 = config_data.pdecoeff.b3;
b4 = config_data.pdecoeff.b4;
c1 = config_data.pdecoeff.c1;
c2 = config_data.pdecoeff.c2;
c3 = config_data.pdecoeff.c3;
incrfact = config_data.pdecoeff.incrfact;

output_path = config_data.gt_path;
% output_path = fullfile(fileparts(git_dir), output_dir);
str_exp = "simulation";
% path_exp = sprintf('%s/src/data/vessel/%s.txt', git_dir, str_exp);
% output_path = sprintf('%s/', output_dir);

wb0 = config_data.parameters.wb0;
wb1 = config_data.parameters.wb1;
wb2 = config_data.parameters.wb2;
wb3 = config_data.parameters.wb3;
wb4 = config_data.parameters.wb4;
wb5 = config_data.parameters.wb5;
wb6 = config_data.parameters.wb6;
wb7 = config_data.parameters.wb7;

wbsys = config_data.parameters.wbsys;
obsindex = config_data.parameters.obsindex;

% Combine the perfusions into an array
obs = [wb0, wb1, wb2, wb3, wb4, wb5, wb6, wb7];
matlab_index=obsindex+1;
wbobs = obs(matlab_index);

lambda = config_data.parameters.lam;
upsilon = config_data.parameters.upsilon;

Xgt1 = config_data.parameters.Xgt1;
Xgt = config_data.parameters.Xgt;
Xw = config_data.parameters.Xw;

oig = config_data.pdecoeff.oig;

om0 = 0;
om1 = 0;
om2 = 0;
om3 = 0;
om4 = 0;
om5 = 0;
om6 = 0;
om7 = 0;

% Call the correct solver based on the number of observers
if nobs == 0
    sol = OneDimBH;
elseif nobs == 1
    sol = OneDimBH_1Obs;
elseif nobs == 3
    sol = OneDimBH_3Obs;
elseif nobs == 4
    sol = OneDimBH_4Obs;
elseif nobs == 8
    sol = OneDimBH_8Obs;
end