
syms b1 b2 b3 b4 'real'

theta30 =0;
theta20=0.8;
theta10 = 0;
a5 = 1.16;
x_gt2 = 0.01;
theta_gt20 = 0.9;

X_gt2 = x_gt2/0.07;

eqs = [b1 + b2 + b3 + b4 == theta10;
b4 == 0.8;
b3 == - a5 * (theta30 - theta20);
b1 * X_gt2^3 + b2 * X_gt2^2 + b3 * X_gt2 + b4 == theta_gt20];
sol = solve(eqs, [b1 b2 b3 b4]);
disp(sol)
 