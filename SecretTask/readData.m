clear;
X=dlmread ('x_train.csv', ";" );
%X=X(2:end,:);


%p=2;
%X_poly = polyFeatures(X, p);
%[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
%X_poly=[ones(size(X,1),1) X_poly];

[X, mu, sigma] = featureNormalize(X);  % Normalize
%X=[ones(size(X,1),1) X];

%treshold = 0.9;
%rc = removeCorrColumns(X,treshold);
%X(:,rc)=[];

m=size(X,1);
n=size(X,2);


y=dlmread ('y_train.csv', ";" );

m_train=round(m*0.7);
%m_train=m;
m_val=m-m_train;

rndIDX = randperm(m); 

X_train = X(rndIDX(1:m_train),:);
X_val = X(rndIDX(m_train+1:end),:);

y_train = y(rndIDX(1:m_train),:);
y_val = y(rndIDX(m_train+1:end),:);


X_test=dlmread ('x_test.csv', ";" );

m_test = size(X_test,1);

%X_test=X_test(2:end,:);
%X_test_poly = polyFeatures(X_test, p);
%[X_test_poly, mu, sigma] = featureNormalize(X_test_poly);  % Normalize
%X_test_poly=[ones(size(X_test_poly,1),1) X_test_poly];

[X_test, mu, sigma] = featureNormalize(X_test);  % Normalize
%X_test=[ones(size(X_test,1),1) X_test];
%X_test(:,rc)=[];