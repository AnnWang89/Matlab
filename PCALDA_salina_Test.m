%function [num_correct,recognition_rate]=PCALDA_salina_Test(GlobalMean,projectPCA,projectLDA,prototype,class01)
function [temp,ID,inc]=PCALDA_salina_Test(GlobalMean,projectPCA,projectLDA,prototype,class01)
principlenum = 50;
ldanum = 5;
inc=0;
projectLDA = projectLDA(:,1:ldanum);%±q20ºû­°¨ì5ºû

[row,col]=size(class01);


%zeromean & projection for testing
ID=[];
    nearindex=1;
    nearEucdis=inf;
for i=1:1:2009
    temp=class01(i,:);
    temp=temp-GlobalMean;%1x204
    temp=temp*projectPCA;%1x20 a row vector
    temp=temp*projectLDA;%1x5 a row vector

    %++++++++ Nearest Neighbor

    for j=1:1:4800
        OAF = temp-prototype(j,:);%Eucdidean distance
        Eucdis=OAF*OAF';%Eucdidean distance
        if nearEucdis > Eucdis
            nearEucdis = Eucdis;
            nearindex=j;
        end
    end
    if ceil(nearindex/300)==1
        inc=inc+1;
    end
    ID = [ID;ceil(nearindex/300)] ;
end

inc=inc/2009;
        
%------------------------



