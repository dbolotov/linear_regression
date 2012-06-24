function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y.
%
%   Function source: ml-class.org

m = length(y); % number of training examples

%Compute the cost of a particular choice of theta
J = (0.5/m) * (X*theta-y)'*(X*theta-y);
end
