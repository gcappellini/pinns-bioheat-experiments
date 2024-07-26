data = readtable('3.csv');

%load franke


x = data(:, 1);
t = data(:, 2);
theta = data(:, 3);

%f = fit([x, t],theta,"poly23")
%plot(f,[x,t],theta)

x_unique = unique(x);
t_unique = unique(t);
[X, T] = meshgrid(x_unique, t_unique);

% Step 4: Interpolate theta values on the grid
Theta = griddata(x, t, theta, X, T);

% Step 5: Plot the surface
figure;
surf(X, T, Theta);
xlabel('X Coordinate');
ylabel('T Coordinate');
zlabel('Theta');
title('Surface Plot of Theta');
colorbar; % Adds a color bar to indicate the value of theta
shading interp; % Improves the visualization by interpolating shading


