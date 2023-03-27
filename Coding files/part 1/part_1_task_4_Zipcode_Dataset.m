clc; 
close all;
clear all;
 
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
for i=1:10
  class_train{i}=train(index_class_train(:,i),1:16);
end
 
prob_class=zeros(1,10);
class_ID=zeros(1,train_length);
D=zeros(1,16);
 
 
for h= .11:.01:.14   
   for i=1:train_length
       for j=1:10
           sum_1=0.0;
           for k=1:length(class_train{j})
              
              D(i,:)=  -1.0*(((test(i,1:16)-class_train{j}(k,:)).^2.0)/(2.0*(h^2.0)));
              sum_1=sum_1+exp(sum(D(i,:)));
           end
           
            prob_class(j)=.1*(sum_1)/(300*2.0*pi*(h^2.0));
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
 
disp('h parameter is');disp(h); 
disp('Test data accuracy')
disp(correct_test/train_length)


end
