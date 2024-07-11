clear all
close all

global K lambda W W1 W2 W3 W4 W5 W6 W7 W8 om1 om2  om3 om4  om5 om6 om7 om8 omega normerr a1 a2 a3 a4 a5 P

a1 = 1.061375;
a2 = 1.9125;
a3 = 6.25e-05;
a4 = 0.7;
a5 = 15.0;
P = 0;

omega=30;

normerr=[]

K=50;
lambda=100;
om1=0;
om2=0;
om3=0;
om4=0;
om5=0;
om6=0;
om7=0;
om8=0;

W=0.67;
W1=0.67;
W2=2;
W3=4;
W4=6;
W5=8;
W6=10;
W7=13;
W8=15;

sol= OneDimBH;
