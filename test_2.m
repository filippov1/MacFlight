clear all

% C1 = [[-3,12,3];[1,3,4];[3,4,3]; [5,6,8]]


R = dlmread('Data/R.dat');
x = dlmread('Data/x.dat');

Rf = R - x*x'
S1 = cholupdate(R, x, '-')

