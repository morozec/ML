%% Initialization
%clear ; close all; clc

%readData;


%% Setup the parameters you will use for this exercise
input_layer_size  = n;  
hidden_layer_size = 100;   % 25 hidden units
num_labels = 5;          % 5 labels, from 0 to 4

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%fprintf('\nChecking Backpropagation... \n');
%  Check gradients by running checkNNGradients
%checkNNGradients;

fprintf('\nTraining Neural Network... \n')

%Определение HighBias-HighVariance
#{
lambda = 1;
lc_m = 100;
[error_train, error_val] = ...
    learningCurve(X_train, y_train, ...
                  X_val, y_val, ...
                  input_layer_size, hidden_layer_size, num_labels, initial_nn_params, ...
                  lambda, lc_m);

plot(1:lc_m, error_train, 1:lc_m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 lc_m 0 40])
#}




%Определение оптимальной lambda
#{
[lambda_vec, error_train, error_val] = ...
    validationCurve(X_train, y_train, X_val, y_val, input_layer_size, hidden_layer_size, num_labels, initial_nn_params);   
   
close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error'); 
#}



%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...                                  
                                   num_labels, X, y, lambda);
                                   
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] =  fmincgNN(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));                
     
	 
pred = predict(Theta1, Theta2, X_test);
%fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_val)) * 100);

% Retrun 5s to 0s
for i=1:m_test
  if pred(i)==5
    pred(i)=0;
  endif
endfor

save result.txt pred;







