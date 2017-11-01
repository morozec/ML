
function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
n = size(X,2);
#disp(m);
K=size(Theta2,1);
         
% You need to return the following variables correctly 
J = 0;
h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');

for i=1:m,
  y_vector = get_y_vector(i,y,K);  
  for k=1:K,
    J=J-y_vector(k)*log(h2(i,k))-(1-y_vector(k))*log(1-h2(i,k));
  end
end
J=J/m;

#–егул€ризаци€
s2=size(Theta1,1);
s3=size(Theta2,1);
s1=size(X,2);
regJ=0;
for j=1:s2,
  for k=2:s1+1,
    regJ=regJ+Theta1(j,k)^2;
  end
end

for j=1:s3,
  for k=2:s2+1,
    regJ=regJ+Theta2(j,k)^2;
  end
end

regJ=regJ*lambda/2/m;
J=J+regJ;


%delta2=zeros(s2+1,1);  %s2 + 1


Delta1 = zeros(size(Theta1,1),size(Theta1,2)); %s2 x (n+1) 
Delta2 = zeros(size(Theta2,1),size(Theta2,2)); %K x (s2+1)

Theta1NoBias= Theta1(:,2:end);  %s2 x n
Theta2NoBias= Theta2(:,2:end);  %K x s2

for t = 1:m,
  delta3=zeros(s3,1);              %K
  y_vector = get_y_vector(t,y,K);  %K
  
  %disp(y_vector);
  %disp('!!!');
  
  a1=X(t,:)';                      %n
  a1=[1; a1];					   %n + 1
  z2=Theta1*a1;                    %s2
  a2=sigmoid(z2);				   %s2	
  a2=[1; a2];                      %s2 + 1
  z3=Theta2*a2;                    %K
  a3=sigmoid(z3);                  %K
  
  for k=1:K,
    delta3(k)=a3(k)-y_vector(k);
  end
   
  
  delta2=(Theta2NoBias')*delta3.* sigmoidGradient(z2); %(s2 x K) * (K x 1) = s2 x 1
  %delta2=delta2(2:end);    
    
  Delta1=Delta1+delta2*(a1)';  % (s2 x 1) * (1x(n+1)) = s2 x (n+1)
  Delta2=Delta2+delta3*(a2)';  % (K x 1) * (1x(s2+1)) = K x (s2+1)
end

Theta1ZeroBias=[zeros(s2,1) Theta1NoBias]; %s2 x (n+1)
Theta2ZeroBias=[zeros(K,1) Theta2NoBias];  %K x (s2+1)

Theta1_grad = Delta1/m + lambda/m*Theta1ZeroBias;
Theta2_grad = Delta2/m + lambda/m*Theta2ZeroBias;

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


function [y_vector]=get_y_vector(i, y, K)    
  y_vector=zeros(K,1);
  yi=y(i);
  y_vector(yi)=1;
end



end
