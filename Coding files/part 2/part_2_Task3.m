clc;
close all
clear all
numb_input=16;
numb_output=1;
number_of_iteration=100;
regularization=10;
size=3000;
filename_train='zipcode_train.csv';
filename_test='zipcode_test.csv';
datatrain=csvread(filename_train);
datatest=csvread(filename_test);


ones2=ones(size,1);
d=zscore(datatrain(:,1:16));
datatrain_train=[ones2 d];

ones2=ones(size,1);
d2=zscore(datatest(:,1:16));
datatest_test=[ones2 d2];

temp=-ones(10,10)+2*eye(10,10);
A=-ones(size,10);
for i=1:1:size
        j=datatrain(i,17);
        A(i,:)=temp(j,:);

end
y=A;
x=datatrain_train;
Auto=transpose(d)*d;
Cross=transpose(d)*y;
weight=inv(Auto)*(Cross);

dmean=0;
dstd=0.01;
w0 = normrnd(dmean,dstd,[numb_input+1 ,numb_output+9]);

%conjugate gradient optimization 
for iteration=1:1:number_of_iteration
    iteration;
    for j=1:1:numb_output+9 
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
        for j=1:1:numb_output+9
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
    est_classID(i)=idx;
   
end

disp('Train Data Accuracy:');

correct=0;
wrong=0;

for i=1:1:size
    if(est_classID(i)==datatrain(i,17))
        correct=correct+1;
    else
        wrong=wrong+1;
        
    end
end

disp('Percentage of Correct');disp(100*correct/3000);
disp('Percentage of Error');disp(100*wrong/3000);


for i=1:size
    yp =datatest_test(i,:)*w0;
    [mn,idx]=max(yp);
    est_classID2(i)=idx;
   
end

disp('Test Data Accuracy:');
correct=0;
wrong=0;

for i=1:1:size
    if(est_classID2(i)==datatest(i,17))
        correct=correct+1;
    else
        wrong=wrong+1;
        
    end
end

disp('Percentage of Test Correct');disp(100*correct/3000);
disp('Percentage of Test Error');disp(100*wrong/3000);