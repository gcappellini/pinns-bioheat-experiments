clear all
close all

global K lambda upsilon W W1 W2 W3 W4 W5 W6 W7 W8 om1 om2  om3 om4  om5 om6 om7 om8 a1 a2 a3 a4 a5 a6 tf

a1 = 67.3;
a2 = 205.2;
a3 = 0.0;
a4 = -2.0;
a5 = 45.0;
a6 = 5.0;

tf = 1;

K=15;
lambda=100;
upsilon=5.0;
om1=0;
om2=0;
om3=0;
om4=0;
om5=0;
om6=0;
om7=0;
om8=0;


W1=0.45;
W2=2*W1;
W3=3*W1;
W4=5*W1;
W5=6*W1;
W6=8*W1;
W7=9*W1;
W8=10*W1;

W=(W3+W4)/2;

sol= OneDimBH;
