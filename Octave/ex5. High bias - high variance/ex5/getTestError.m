function [error_test] = ...
    getTestError(X_poly, y, X_poly_test, ytest, lambda)
	
[theta] = trainLinearReg(X_poly, y, lambda);
[J_test, grad] = linearRegCostFunction(X_poly_test, ytest, theta, 0);
error_test=J_test;

end