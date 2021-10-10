%function [GlobalMean,FACE,projectPCA,prototypeFACE,eigvector]=PCALDA_Face_Train()

function [index,eigvalue,GlobalMean,projectPCA,prototypeFACE,eigvector,projectLDA,pcaTotal,PeopleMean,SW,SB,FACE]=PCALDA_Face_Train();

people = 40;

withinsample = 5;%�C�Ӹ�ƨ��X��

principlenum = 50;%��������50��

FACE = [];%�s�QŪ�X�Ӫ����

for k = 1:1:people
    
    for m=1:2:10
        matchstring=['ORL3232' '\' num2str(k) '\' num2str(m) '.bmp'];%num2str��̭���number ��string
        matchX=imread(matchstring);%matchX�����ɼƦr�L�k�p��
        matchX=double(matchX);%�אּ�i�p��Ʀr�CmatchX���@�ӯx�}32*32
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
 
eigvalue=diag(latent);  % extract the diagnal only(�ӡ�????)
 
[junk,index]= sort(eigvalue,'descend');
 
pca1=pca(:,index);
 
eigvalue=eigvalue(index);
projectPCA=pca1(:,1:50); % extract the principle component
%projectPCA?�X????��x�X
%++++++++++++++++++++++++LDA transform ++++++++++++++++++++++
pcaTotal =[];

for i=1:1:FACERow
    temp = FACE(i,:);
    temp = temp*projectPCA;
    pcaTotal = [pcaTotal;temp];
end

for i=1:withinsample:withinsample*people
    within=pcaTotal(i:i+withinsample-1,:);%�Ȧs��@���OPCA�Ŷ����V�m�v��
    if(i==1)
        PeopleMean=mean(within);
        SW=within'*within; %SW=cov(within)
        %�C�ӳ��h��mean,�M���SW
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

