function [t2,input,hiddenmatrix]=BP2;       %function approximation 
input=[];
target=[];
out=[];             %a
s=[];
t=[];               
x1=[];              %dimension 1
x2=[];              %dimension 2
x3=[];              %dimension 3
x4=[];              %dimension 4

for i=1:1:600       %generate the data(including training &testing)
    x1=rand;
    x2=rand;
    x3=rand;
    x4=rand;
    s=[x1,x2,x3,x4];
    input=[input;s];
    t=0.8*x1*x2*x3*x4+x1.^2+x2.^2+x3.^3+x4.^2+x2*0.7-x2.^2*x3.^2+0.5*x1*x4.^2+x4*x2.^3-(x1*x2+x1*x2*x3*x4).^3+(x1-x2+x3-x4)+(x1*x4)-(x2*x3)-2;
    target=[target;t];
end

outnet=[];

%initialize the weight matrix
outputmatrix=zeros(2,1);
for i=1:1:2
    for j=1:1:1
        outputmatrix(i,j)=rand;
    end
end

hiddenmatrix=zeros(4,2);

for i=1:1:4
    for j=1:1:2
        hiddenmatrix(i,j)=rand;
    end
end

RMSE1=zeros(100,1);     %store the error
RMSE2=zeros(100,1);

%training
for epoch=1:1:100
    t1=[];
    t2=[];
    for iter=1:1:400
        %forward �e�ǳ���
        %training
        hiddensigma=input(iter,:)*hiddenmatrix;
        hiddennet=logsig(hiddensigma);
        outputsigma=hiddennet*outputmatrix;
        outputnet=purelin(outputsigma);
        %simalation
        if iter+400<=600% take the first 400 as training samples,the remaining 200 as simulations
            hsigma=input(iter+400,:)*hiddenmatrix;
            hnet=logsig(hsigma);
            osigma=hnet*outputmatrix;
            onet=purelin(osigma);
            mis=target(iter+400)-onet;
            t2=[t2;mis.^2];
        end
        %backward part �˶ǳ���
        %delta of outputmatrix ��X�h�� delta
        doutputnet=dpurelin(outputsigma);
        deltaoutput=(target(iter)-outputnet)*doutputnet;
        error=target(iter)-outputnet;
        t1=[t1;error.^2];
        %delta of hidden layer ���üh��delta
        tempdelta=deltaoutput*outputmatrix;
        transfer=dlogsig(hiddensigma,logsig(hiddensigma));
        deltahidden=[];
        for i=1:1:2
            deltahidden=[deltahidden;tempdelta(i)*transfer(i)];
        end
        %output layer weight update ��X�h�v����s
        newoutputmatrix=outputmatrix+0.25*(deltaoutput*hiddennet)';
        outputmatrix=newoutputmatrix;
        %hidden layer ���üh�v����s 
        newhiddenmatrix=hiddenmatrix;
        for i=1:1:2
            for j=1:1:4
                newhiddenmatrix(j,i)=hiddenmatrix(j,i)+0.25*deltahidden(i)*input(iter,j);
            end
        end
        hiddenmatrix=newhiddenmatrix;
    end
        RMSE1(epoch)=sqrt(sum(t1)/400);
        RMSE2(epoch)=sqrt(sum(t2)/400);
        fprintf('epoch %.0f:  RMSE =%.3f\n',epoch,sqrt(sum(t1)/400));
end
fprintf('\nTotal number of epochs: %g\n',epoch);
    fprintf('Final RMSE: %g\n',RMSE1(epoch));
    figure(1);
    plot(1:epoch,RMSE1(1:epoch),1:epoch,RMSE2(1:epoch));
    legend('Training', 'Simulation');
    ylabel('RMSE');
    xlabel('Epoch');
%%%
Train_Correct=0;%transform the function approximation
for i=1:400     %i
    hiddensigma=input(i,:)*hiddenmatrix;
    hiddennet=logsig(hiddensigma);
    outputsigma=hiddennet*outputmatrix;
    outputnet=purelin(outputsigma);
    out=[out;outputnet];
    if outputnet > target(i)-0.5 && outputnet <=target(i)+0.5
        Train_Correct=Train_Correct+1;
    end
end

Simu_Correct=0;%transform the function approximation
for i=401:length(input)     %i
    hiddensigma=input(i,:)*hiddenmatrix;
    hiddennet=logsig(hiddensigma);
    outputsigma=hiddennet*outputmatrix;
    outputnet=purelin(outputsigma);
    outnet=[outnet;outputnet];
    if outputnet > target(i)-0.5 && outputnet <=target(i)+0.5
        Simu_Correct=Simu_Correct+1;
    end
end
figure(2);
plot(401:length(input),target(401:length(input)),401:length(input),(1:200));
legend('Function','Simulation');
Train_Percent=(Train_Correct)/400;
Simu_Percent=(Simu_Correct)/(length(input)-400);
Train_correct_percent=Train_Percent;
Simu_correct_percent=Simu_Percent;
figure(3);
[m,b,r]=postreg(out',target(1:400)');