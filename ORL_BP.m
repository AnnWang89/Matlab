function [input,target]=ORL_BP(prototypeFACE);
%------------------------------
%train_BP
%------------------------------
input = prototypeFACE;
target =[];
for i=1:1:100
    target=[target;ceil(i/5)];
end

% initialize the weight matrix
outnet=[];

% initialize the weight matrix
outputmatrix=zeros(35,1);%35為神經元數量
for i=1:1:35
 for j=1:1:1
   outputmatrix(i,j)=rand;
 end
end

hiddenmatrix=zeros(20,35)
for i=1:1:20
 for j=1:1:35
   hiddenmatrix(i,j)=rand;
 end
end


RMSE1=zeros(600,1);
RMSE2=zeros(600,1);


% Training
for epoch=1:1:600
t1=[];
t2=[];
for iter=1:1:100

% forward 前傳部分

% training
hiddensigma=input(iter,:)*hiddenmatrix;
hiddennet=logsig(hiddensigma);       
outputsigma=hiddennet*outputmatrix;
outputnet=purelin(outputsigma);    


% simalation 
%if iter+400<=600 % take the first 400 as training samples, the remaining 200 as simulations
%hsigma=input(iter+400,:)*hiddenmatrix;
%hnet=logsig(hsigma);       

%osigma=hnet*outputmatrix;
%onet=purelin(osigma);

%mis=target(iter+400)-onet;
%t2=[t2;mis.^2];
%end





% backward part 倒傳部分
% delta of outputmatrix 輸出層的 delta
doutputnet=dpurelin(outputsigma);
deltaoutput=(target(iter)-outputnet)*doutputnet;
error=target(iter)-outputnet;
t1=[t1;error.^2];


% delta of hidden layer 隱藏層的 delta
tempdelta=deltaoutput*outputmatrix;
transfer=dlogsig(hiddensigma,logsig(hiddensigma));
deltahidden=[];
for i=1:1:35
deltahidden=[deltahidden;tempdelta(i)*transfer(i)];
end

% output layer weight update 輸出層權重更新
newoutputmatrix=outputmatrix+0.025*(deltaoutput*hiddennet)';
outputmatrix=newoutputmatrix;

% hidden layer 隱藏層權重更新
newhiddenmatrix=hiddenmatrix;
for i=1:1:35
for j=1:1:20
newhiddenmatrix(j,i)=hiddenmatrix(j,i)+0.025*deltahidden(i)*input(iter,j);
end
end
hiddenmatrix=newhiddenmatrix;    
end


RMSE1(epoch) = sqrt(sum(t1)/100);
RMSE2(epoch) = sqrt(sum(t2)/100);

fprintf('epoch %.0f:  RMSE = %.3f\n',epoch, sqrt(sum(t1)/100));
end


fprintf('\nTotal number of epochs: %g\n', epoch);
fprintf('Final RMSE: %g\n', RMSE1(epoch));
figure(1);
plot(1:epoch,RMSE1(1:epoch),1:epoch,RMSE2(1:epoch));
legend('Training','Simulation');
ylabel('RMSE');xlabel('Epoch');



Train_Correct=0;

for i=1:100
    
    hiddensigma=input(i,:)*hiddenmatrix;
    hiddennet=logsig(hiddensigma);       
    outputsigma=hiddennet*outputmatrix;
    outputnet=purelin(outputsigma);
    out=[out;outputnet];
        if outputnet > target(i)-0.5 &  outputnet <= target(i)+0.5%在目標值上下0.5內為正確
            Train_Correct=Train_Correct+ 1;
        end
end


Simu_Correct=0;

for i=101:length(input)
    
    hiddensigma=input(i,:)*hiddenmatrix;
    hiddennet=logsig(hiddensigma);       
    outputsigma=hiddennet*outputmatrix;
    outputnet=purelin(outputsigma);
    outnet=[outnet;outputnet];
        if outputnet > target(i)-0.5 &  outputnet <= target(i)+0.5
            Simu_Correct=Simu_Correct+ 1;
        end
end
figure(2);
plot(101:length(input),target(101:length(input)),101:length(input),(1:100))
legend('Function','Simulation');
Train_Percent= (Train_Correct) / 100;
Simu_Percent= (Simu_Correct) / (length(input)-100);
Train_correct_percent=Train_Percent
Simu_correct_percent=Simu_Percent
figure(3)
[m,b,r]=postreg(out',target(1:100)');


%------------------------------
% initialize the weight matrix
outputmatrix=zeros(35,1);%35為神經元數量
for i=1:1:35
 for j=1:1:1
   outputmatrix(i,j)=rand;
 end
end

hiddenmatrix=zeros(20,35);
for i=1:1:20
 for j=1:1:35
   hiddenmatrix(i,j)=rand;
 end
end


RMSE1=zeros(600,1);
RMSE2=zeros(600,1);


% Training
for epoch=1:1:600
t1=[];
t2=[];
for iter=1:1:100

% forward 前傳部分

% training
hiddensigma=input(iter,:)*hiddenmatrix;
hiddennet=logsig(hiddensigma);       
outputsigma=hiddennet*outputmatrix;
outputnet=purelin(outputsigma);    

% backward part 倒傳部分
% delta of outputmatrix 輸出層的 delta
doutputnet=dpurelin(outputsigma);
deltaoutput=(target(iter)-outputnet)*doutputnet;
error=target(iter)-outputnet;
t1=[t1;error.^2];


% delta of hidden layer 隱藏層的 delta
tempdelta=deltaoutput*outputmatrix;
transfer=dlogsig(hiddensigma,logsig(hiddensigma));
deltahidden=[];
for i=1:1:35
deltahidden=[deltahidden;tempdelta(i)*transfer(i)];
end

% output layer weight update 輸出層權重更新
newoutputmatrix=outputmatrix+0.025*(deltaoutput*hiddennet)';
outputmatrix=newoutputmatrix;

% hidden layer 隱藏層權重更新
newhiddenmatrix=hiddenmatrix;
for i=1:1:35
for j=1:1:20
newhiddenmatrix(j,i)=hiddenmatrix(j,i)+0.025*deltahidden(i)*input(iter,j);
end
end
hiddenmatrix=newhiddenmatrix;    
end


RMSE1(epoch) = sqrt(sum(t1)/100);
RMSE2(epoch) = sqrt(sum(t2)/100);

fprintf('epoch %.0f:  RMSE = %.3f\n',epoch, sqrt(sum(t1)/100));
end


fprintf('\nTotal number of epochs: %g\n', epoch);
fprintf('Final RMSE: %g\n', RMSE1(epoch));
figure(1);
plot(1:epoch,RMSE1(1:epoch),1:epoch,RMSE2(1:epoch));
legend('Training','Simulation');
ylabel('RMSE');xlabel('Epoch');



Train_Correct=0;

for i=1:100
    
    hiddensigma=input(i,:)*hiddenmatrix;
    hiddennet=logsig(hiddensigma);       
    outputsigma=hiddennet*outputmatrix;
    outputnet=purelin(outputsigma);
    out=[out;outputnet];
        if outputnet > target(i)-0.5 &  outputnet <= target(i)+0.5%在目標值上下0.5內為正確
            Train_Correct=Train_Correct+ 1;
        end
end

Train_Percent= (Train_Correct) / 100;
Train_correct_percent=Train_Percent
