%function [GlobalMean,FACE,projectPCA,prototypeFACE,eigvector]=PCALDA_Face_Train()

function [index,eigvalue,GlobalMean,projectPCA,prototypeFACE,eigvector,projectLDA,pcaTotal,PeopleMean,SW,SB,FACE]=PCALDA_Face_Train();

people = 40;

withinsample = 5;%每個資料取幾筆

principlenum = 50;%降維降到50維

FACE = [];%存被讀出來的資料

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
projectPCA=pca1(:,1:50); % extract the principle component
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

prototypeFACE=pcaTotal*projectLDA(:,1:20); 

