function [nn_params] = trainNN(X, y, input_layer_size, hidden_layer_size, num_labels, initial_nn_params, lambda)
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
        
% Now, costFunction is a function that takes in only one argument
options = optimset('MaxIter', 100, 'GradObj', 'on');
        
[nn_params, cost] =  fmincgNN(costFunction, initial_nn_params, options);


end
