 
clc
clear all
close all
in=8;
out=7;
trn=csvread('regression_train.csv');
tr_in1=trn(:,1:8);
one=ones(1768,1);
tr_in=[one tr_in1];
tr_out=trn(:,9:15);
R=transpose(tr_in)*tr_in;
C=transpose(tr_in)*tr_out;
w=R\C;

%Training error
ind_error=0;
Y_out=tr_in*w;
for j=1:1:7
    error_f=0;
for i=1:1:1768
    error_f=error_f+(tr_out(i,j)-Y_out(i,j))^2;
end
ind_error(j)=error_f/1768;
end
disp('Training Error:')
disp(ind_error')

% testing error
tst=csvread('regression_tst.csv');
tst_in1=tst(:,1:8);
one=ones(1000,1);
tst_in=[one tst_in1];
tst_out=tst(:,9:15);
Y_tst_out=tst_in*w;

error1=(Y_tst_out-tst_out).^2/1000;
ind_error1=0;
for j=1:1:7
    error_f=0;
for i=1:1:1000
    error_f=error_f+error1(i,j);
end
ind_error1(j)=error_f;
end

disp('Testing Error:')
disp(ind_error1')



