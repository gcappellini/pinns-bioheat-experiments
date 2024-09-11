clear all
close all

global K lambda delta upsilon W W0 W1 W2 W3 W4 W5 W6 W7 theta_w theta1 theta20 om0 om1 om2  om3 om4  om5 om6 om7 a1 a2 a3



%function [a1, a2, a3, delta, lambda, upsilon, W0, W1, W2, W3, W4, W5, W6, W7, theta_w, theta1, K] = loadProperties(filename)
% Check if filename is provided
%if nargin < 1
    filename = 'properties.json'; % Default filename
%end

% Read JSON file
try
    jsonData = fileread(filename);  % Read the JSON file as a string
catch
    error('Error reading the JSON file. Make sure the file exists and the path is correct.');
end

% Parse JSON data
try
    data = jsondecode(jsonData);    % Decode JSON data into a MATLAB struct
catch
    error('Error parsing JSON file. Ensure it is correctly formatted.');
end

% Extract parameters from the struct
try
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
    W0 = data.W0;
    W1 = data.W1;
    W2 = data.W2;
    W3 = data.W3;
    W4 = data.W4;
    W5 = data.W5;
    W6 = data.W6;
    W7 = data.W7;
    delta = data.delta;
    lambda = data.lambda;
    upsilon = data.upsilon;

catch
    error('Error accessing JSON fields. Ensure JSON contains required fields.');
end


% Compute constants a1, a2, a3
a1 = (L0^2/tauf)*(rho*cp/k);
a2 = L0^2*cp/k;
a3 = (h*L0)/k;
theta_w = (t_w - t_room)/(t_max - t_room);
theta20 = (t_y20 - t_room)/(t_max - t_room);
theta1 = 0;

om0=0;
om1=0;
om2=0;
om3=0;
om4=0;
om5=0;
om6=0;
om7=0;


W=(W3+W4)/2;

sol= OneDimBH;