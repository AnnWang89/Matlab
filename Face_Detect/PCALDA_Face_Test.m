%[temp,ID,inc,FACE]=PCALDA_Face_Test(GlobalMean,projectPCA,eigvector,prototypeFACE);
function [projectLDA,temp,ID,inc,FACE,rate]=PCALDA_Face_Test(GlobalMean,projectPCA,eigvector,projectLDA,prototypeFACE);

people = 40;

withinsample = 5;%每個資料取幾筆
principlenum = 50;%降維降到50維
inc=0;
FACE = [];%存被讀出來的資料
projectLDA=projectLDA(:,1:20);%降維降到50維
for k = 1:1:people
    
    for m=2:2:10
        matchstring=['ORL3232' '\' num2str(k) '\' num2str(m) '.bmp'];%num2str把裡面的number 變string
        matchX=imread(matchstring);%matchX為圖檔數字無法計算
        matchX=double(matchX);%改為可計算數字。matchX維一個矩陣32*32
        if (k==1 && m==2)
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
[FACERow,col]=size(FACE); 
%zeromean & projection for testing
ID=[];
for i=1:1:FACERow
    nearindex=0;
    nearEucdis=inf;
    temp=FACE(i,:);
    temp=temp-GlobalMean;%1x204
    temp=temp*projectPCA;%1x20 a row vector
    temp=temp*projectLDA;
    %++++++++ Nearest Neighbor
    for j=1:1:200
        OAF = temp-prototypeFACE(j,:);%Eucdidean distance
        Eucdis=OAF*OAF';%Eucdidean distance
        if nearEucdis > Eucdis
            nearEucdis = Eucdis;
            nearindex=j;
        end
    end
    if ceil(nearindex/5)==ceil(i/5)
        inc=inc+1;
    end
    ID = [ID;ceil(nearindex/5)] ;
end

rate = inc/200
%inc=inc/(withinsample*principlenum);        
%------------------------
