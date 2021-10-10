function  [projectPCA,prototypeFACE,projectLDA,out,outputnet,inputtrain,t1]=ORL();
people = 20;

withinsample = 5;%每個資料取幾筆
traintimes =600;
principlenum = 70;%降維降到50維
LDAdim=5;
snum=85;%神經元
FACE = [];%存被讀出來的資料
%------------------------------
%train_PCALDA
%------------------------------
%讀資料
for k = 1:1:people
    for m=1:2:10
        matchstring=['ORL3232' '\' num2str(k) '\' num2str(m) '.bmp'];%num2str把裡面的number 變string
        matchX=imread(matchstring);%matchX為圖檔數字無法計算
        matchX=double(matchX);%改為可計算數字。matchX維一個矩陣32*32
        if (k==1 && m==1)
            [row,col]=size(matchX);
        end
        matchtempF=[];
        %--arrange the image into a vector
        for n=1:1:row
            matchtempF=[matchtempF,matchX(n,:)];
        end
        FACE=[FACE;matchtempF];
    end
end
FACE_2=[];
for k = 1:1:people
    
    for m=2:2:10
        matchstring=['ORL3232' '\' num2str(k) '\' num2str(m) '.bmp'];%num2str把裡面的number 變string
        matchX=imread(matchstring);%matchX為圖檔數字無法計算
        matchX=double(matchX);%改為可計算數字。matchX維一個矩陣32*32
        if (k==1 && m==2)
            [row,col]=size(matchX);
        end
        matchtempF_2=[];
        %--arrange the image into a vector
        for n=1:1:row
            matchtempF_2=[matchtempF_2,matchX(n,:)];
        end
        FACE_2=[FACE_2;matchtempF_2];
    end
end
 
[FACERow,FACECol]=size(FACE);
GlobalMean=mean(FACE);
 
%zeromean
for i=1:1:FACERow
    FACE(i,:) = FACE(i,:)-GlobalMean;
end
%---------------------------------
 
%SST cov
SST = FACE'*FACE;
%--------------------------------
 
[pca,latent] = eig(SST);
 
eigvalue=diag(latent);  % extract the diagnal only(±×????)
 
[junk,index]= sort(eigvalue,'descend');
 
pca1=pca(:,index);
 
eigvalue=eigvalue(index);
projectPCA=pca1(:,1:principlenum); % extract the principle component
%projectPCA?°????¯x°
%++++++++++++++++++++++++LDA transform ++++++++++++++++++++++
pcaTotal =[];

for i=1:1:FACERow
    temp = FACE(i,:);
    temp = temp*projectPCA;
    pcaTotal = [pcaTotal;temp];
end

for i=1:withinsample:withinsample*people
    within=pcaTotal(i:i+withinsample-1,:);%暫存單一類別PCA空間中訓練影像
    if(i==1)
        PeopleMean=mean(within);
        SW=within'*within; %SW=cov(within)
        %每個都去撿mean,然後算SW
    end
    if(i>1)
        SW=SW+within'*within;    %SW=SW+cov(within)
        PeopleMean=[PeopleMean;mean(within)];  %this matrix is for Between
    end
end      %end of i
%--------------------------
SB=PeopleMean'*PeopleMean;%SB=cov(ClassMean)

[eigvector,eigvalue]=eig(inv(SW)*SB);    %----> arg max (SB/SW)
eigvalue=diag(eigvalue);    
[junk,index]=sort(eigvalue,'descend');
eigvalue =eigvalue(index);
projectLDA=eigvector(:,index);

prototypeFACE=pcaTotal*projectLDA(:,1:LDAdim); 
%projectLDA=projectLDA(:,1:20);%降維降250維

inputtrain=[];
projectLDA=projectLDA(:,1:LDAdim);
for i=1:1:FACERow
    temp=FACE_2(i,:);
    temp=temp-GlobalMean;%1x204
    temp=temp*projectPCA;%1x20 a row vector
    temp=temp*projectLDA;
    inputtrain=[inputtrain;temp];
end
%------------------------------
%train_BP
%------------------------------
input = [prototypeFACE;inputtrain];
target =[];
out =[];
for i=1:1:100
    target=[target;ceil(i/5)];
end
target=[target;target];
outnet=[];

% initialize the weight matrix
outputmatrix=zeros(snum,1);%snum為神經元數量
for i=1:1:snum
 for j=1:1:1
   outputmatrix(i,j)=rand;
 end
end


hiddenmatrix=zeros(LDAdim,snum)
for i=1:1:LDAdim
 for j=1:1:snum
   hiddenmatrix(i,j)=rand;
 end
end


RMSE1=zeros(traintimes,1);
RMSE2=zeros(traintimes,1);


% Training
for epoch=1:1:traintimes
t1=[];
t2=[];
for iter=1:1:100

% forward 前傳部分

% training
hiddensigma=input(iter,:)*hiddenmatrix;
hiddennet=logsig(hiddensigma);       
outputsigma=hiddennet*outputmatrix;
outputnet=purelin(outputsigma) ;  


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
for i=1:1:snum
deltahidden=[deltahidden;tempdelta(i)*transfer(i)];
end

% output layer weight update 輸出層權重更新
newoutputmatrix=outputmatrix+0.025*(deltaoutput*hiddennet)';
outputmatrix=newoutputmatrix;

% hidden layer 隱藏層權重更新
newhiddenmatrix=hiddenmatrix;
for i=1:1:snum
for j=1:1:LDAdim
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
