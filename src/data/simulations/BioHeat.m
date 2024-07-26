clear all
close all

global K lambda upsilon W W0 W1 W2 W3 W4 W5 W6 W7  om0 om1 om2  om3 om4  om5 om6 om7 a1 a2 a3 a4 a5 a6 tf

a1 = 87.083;
a2 = 156.75;
a3 = 0.0847466;
a4 = -2.0;
a5 = 30.0;
a6 = 3.4703631;

tf = 1;

K=15;
lambda=10;
upsilon=5.0;
om0=0;
om1=0;
om2=0;
om3=0;
om4=0;
om5=0;
om6=0;
om7=0;


W0=1;
W1=2;
W2=3;
W3=5;
W4=6;
W5=8;
W6=9;
W7=10;

W=W0;

sol= OneDimBH;
