clear all
close all

global K lambda W W1 W2 W3 W4 W5 W6 W7 W8 om1 om2  om3 om4  om5 om6 om7 om8 a1 a2 a3 a4 a5 a6 P0

a1 = 1.061375;
a2 = 1.9125;
a3 = 6.25e-05;
a4 = 0.7;
a5 = 15.0;
a6 = 0.1666667;
P0 = 1e+05;


K=4;
lambda=100;
om1=0;
om2=0;
om3=0;
om4=0;
om5=0;
om6=0;
om7=0;
om8=0;

W=0.45;
W1=W;
W2=2*W;
W3=3*W;
W4=5*W;
W5=6*W;
W6=8*W;
W7=9*W;
W8=10*W;

sol= OneDimBH;
