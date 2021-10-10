function [w,num_incorrect,b,p]=perceptron_Practice()
%t為目標值
p=[1,0;1,1;0,1;0,0];
t=[1;0;1;0];
w=[0.5,-0.5;0.1,-0.1];

b=[0.5;0.1];%bias
num_incorrect=0;
for i=1:1:4
    for j=1:1:2
        temp=(p(i,:)*w(j,:)'-b(j));%不知道有沒有-b
        if(temp>0)
            a(j)=1;
        else
            a(j)=0;
        end
        w(j,:)=w(j,:)+((t(i)-a(j)))*p(i,:);
        b(j)=b(j)+(t(i)-a(j));
    end
end

for i=1:1:4
    for j=1:1:2
        temp=(p(i,:)*w(j,:)'-b(j));%不知道有沒有-b
        if(temp>0)
            a(j)=1
        else
            a(j)=0
        end
    end
    
    if( t(i) ~= xor(a(1),a(2)) )
        num_incorrect=num_incorrect+1
    end
end
%while num_incorrect ~= 0 
 %   num_incorrect=0;
  %  for i=1:1:3
   %     temp=(p(i,:)*w');
    %    if(temp>0)
     %        a=1
      %  else
       %     a=0
        %end
       % w=w+((t(i)-a))*p(i,:);        
       % if( t(i) ~= a)
       %     num_incorrect=num_incorrect+1;
       % end
   % end
%end
%nb=p(1,:)*w'+b;