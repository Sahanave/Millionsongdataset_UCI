%Proceed to use if you wish to understand how to calculate the covariance
%matrices and how i arrived at the feature plots%
%%%%this is a matlab code.Run it in Matlab********
clear all
%plotting the covariance matrix using matlab
dataset=load('raw_data.txt');
[a b]=size(dataset);
data=zscore(dataset);
cov=data'*data;
cov=cov/515345;
figure(1)
imagesc(cov)
colorbar
title('covariance matrix for 90 attributes');
data=data(:,1:12);
cov=data'*data;
cov=cov/515345;
figure(2)
imagesc(cov)
colorbar
title('covariance matrix for 12 attributes');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X=zeros(90,12);
count=zeros(1,90);
for j=1:a
    for i=1922:1:2011
        if(dataset(j,1)==i)
            X(i-1921,:)=dataset(j,2:13);
            count(i-1921)=count(i-1921)+1;
        end
    end
end
for i=1:90
    X(i,:)=X(i,:)/count(i);
end
X=[X(1,:);X(3:90,:)];
X=zscore(X);
y=[1922,1924:2011];
figure(3)
plot(y,X(:,1));
title('the graph for loudness')
figure(4)
plot(y,X(:,2));
title('the graph for feature 2')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%this is a matlab code.Run it in Matlab The matlab part ends here ********
    
    


