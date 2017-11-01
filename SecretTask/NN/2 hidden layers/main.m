%% Initialization
clear ; close all; clc

readData;


%% Setup the parameters you will use for this exercise
input_layer_size  = n;  
hidden_layer_size1 = 100;   % 100 hidden units
hidden_layer_size2 = 25;   % 25 hidden units
num_labels = 5;          % 5 labels, from 0 to 4

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size1);
initial_Theta2 = randInitializeWeights(hidden_layer_size1, hidden_layer_size2);
initial_Theta3 = randInitializeWeights(hidden_layer_size2, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)];

%fprintf('\nChecking Backpropagation... \n');
%  Check gradients by running checkNNGradients
%checkNNGradients;

fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 10);

%  You should also try different values of lambda
lambda = 0;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size1, ...
                                   hidden_layer_size2, ...
                                   num_labels, X, y, lambda);
                                   
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] =  fmincgNN(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size1 * (input_layer_size + 1)), ...
                 hidden_layer_size1, (input_layer_size + 1));
                 
Theta2 = reshape(nn_params((1 + (hidden_layer_size1 * (input_layer_size + 1))): ...
        (1 + (hidden_layer_size1 * (input_layer_size + 1)) + hidden_layer_size2 * (hidden_layer_size1 + 1))),...
        hidden_layer_size2, (hidden_layer_size1 + 1));

Theta3 = reshape(nn_params((1 + 1 + (hidden_layer_size1 * (input_layer_size + 1)) + hidden_layer_size2 * (hidden_layer_size1 + 1)):end), ...
                 num_labels, (hidden_layer_size2 + 1));
           
	 
pred = predict(Theta1, Theta2, Theta3, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

save result.txt pred;

