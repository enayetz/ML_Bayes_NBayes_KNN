clc;
clear all;
close all;
numb_input=2;
numb_output=1;
regularization=0.000;
number_of_iteration=500;

filename_train='generated_train.csv';
filename_test='generated_test.csv';
datatrain=csvread(filename_train);
datatest=csvread(filename_test);
size=length(datatrain)
%data for test and train
ones2=ones(size,1);
d=zscore(datatrain(:,1:2));
datatrain_train=[ones2 d];

ones2=ones(size,1);
d2=zscore(datatest(:,1:2));
datatest_test=[ones2 d2];

temp=-ones(2,2)+2*eye(2,2);
A=-ones(size,2);
for i=1:1:size
    if (datatrain(i,3)==1)
        A(i,:)=temp(1,:);
    elseif (datatrain(i,3)==2)
         A(i,:)=temp(2,:);
    else
        disp('error')
    end  
end


y=A;
x=datatrain_train;
Auto=transpose(d)*d;
Cross=transpose(d)*y;
weight=inv(Auto)*(Cross);
%random weight
dmean=0;
dstd=0.01;
w0 = normrnd(dmean,dstd,[numb_input+1 ,numb_output+1]);
%conjugate gradient optimization

for iteration=1:1:number_of_iteration
    iteration;
    for j=1:1:numb_output+1 
    ytn=y(:,j);
    error_sum=0;
        for n=1:1:size 
        xn=x(n,:);
        tp=0;
            for i=1:1:numb_input+1 
            pred=xn(i)*w0(i,j);
            tp=tp+pred;
            end
        difference=ytn(n) - tp;
        error_sum=error_sum+difference*difference;
        end
    Error(j) = double((1/(2*size))*error_sum); 

    end

    for k=1:numb_input+1
        for j=1:1:numb_output+1 
        tng=y(:,j);
        grad_sum=0;
            for n=1:1:size
                xn=x(n,:);
                tp=0;
                for i=1:1:numb_input+1  
                    pred=xn(i)*w0(i,j);
                    tp=tp+pred;
                end
                difference=tng(n) - tp;
                gradn = -(difference * x(n,k)); 
                grad_sum=grad_sum + gradn;
            end
        grad(k,j) = ((1/size)*grad_sum)+regularization*w0(k,j); 
        end
    end
    w0 = w0 - 0.01*grad;

    error_norm(iteration)=norm(Error,2);
    grad_norm(iteration)=norm(grad,2);
    w0;
    if((error_norm(iteration)>grad_norm(iteration)))
        break
    end
end


for i=1:size
    yp =datatrain_train(i,:)*w0;
    [mn,idx]=max(yp);
    EST_classID(i)=idx;
   
end

disp('train data accuracy:');

correct=0;
wrong=0;

for i=1:1:400
    if(EST_classID(i)==datatrain(i,3))
        correct=correct+1;
    else
        wrong=wrong+1;
        
    end
end

disp('percentage of correct');disp(100*correct/400);
disp('percentage of error');disp(100*wrong/400);


for i=1:size
    yp =datatest_test(i,:)*w0;
    [mn,idx]=max(yp);
    estID_classID2(i)=idx;
   
end

disp('test data accuracy:');
correct=0;
wrong=0;

for i=1:1:400
    if(estID_classID2(i)==datatest(i,3))
        correct=correct+1;
    else
        wrong=wrong+1;
        
    end
end

disp('percentage of test correct');disp(100*correct/400);
disp('percentage of test error');disp(100*wrong/400);