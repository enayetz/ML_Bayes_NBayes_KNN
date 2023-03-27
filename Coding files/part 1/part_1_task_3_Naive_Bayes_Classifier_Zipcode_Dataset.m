clc 
close all
TRN = load('training_zipcode.mat');
TST = load('testing_zipcode.mat');

train = TRN.data;
test= TST.data;

index_class_train=zeros(300,10);
train_length=size(train,1);

for i=1:10    
 index_class_train(:,i)= find(train(:,17)==i);
end

class_length= size(index_class_train,1);
class_train=cell(1,10);
mean_train=cell(1,10);
cov_train=cell(1,10);

for i=1:10
class_train{i}=train(index_class_train(:,i),1:16);
mean_train{i}= mean(class_train{i},1);
z=3.5*eye(300,16)+class_train{i}; % inducing error
cov_train{i}=var(z,1);
end
%% test data accuracy
prob_class=zeros(1,10);
class_ID=zeros(1,train_length);
D=zeros(1,16);
    
for i=1:train_length
    for j=1:10
       D(i,:)=  -.5*(((test(i,1:16)-mean_train{j}).^2)./cov_train{j});
       prob_class(j)=.1*exp(sum(D(i,:)))/(2*pi*sqrt(prod(cov_train{j})));
    end 
    [val,idx]=max( prob_class);
    class_ID(i)= idx;
end

correct_test=0.0;

for i=1:train_length
    if class_ID(i)== test(i,17)
        correct_test=correct_test+1;
    end
end


disp('Test data accuracy')
disp(correct_test*100/train_length)

%% train data accuracy

prob_class_train=zeros(1,10);
class_ID=zeros(1,train_length);
D=zeros(1,16);
    
for i=1:train_length
    for j=1:10
       D(i,:)=  -.5*(((train(i,1:16)-mean_train{j}).^2)./cov_train{j});
       prob_class_train(j)=.1*exp(sum(D(i,:)))/(2*pi*sqrt(prod(cov_train{j})));
    end 
    [val,idx]=max( prob_class_train);
    class_ID(i)= idx;
end

correct_train=0.0;

for i=1:train_length
    if class_ID(i)== train(i,17)
        correct_train=correct_train+1;
    end
end

disp('Train data accuracy')
disp(correct_train*100/train_length)