function [Theta1, Theta2] = trainLinearReg(X, y, input_layer_size, hidden_layer_size, num_labels, lambda)
%TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
%regularization parameter lambda
%   [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
%   the dataset (X, y) and regularization parameter lambda. Returns the
%   trained parameters theta.
%


% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...                                  
                                   num_labels, X, y, lambda);
                                   
[nn_params, cost] =  fmincgNN(costFunction, initial_nn_params, options);

% Now, costFunction is a function that takes in only one argument
options = optimset('MaxIter', 20, 'GradObj', 'on');

% Minimize using fmincg
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

end
