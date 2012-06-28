%Linear Regression with Gradient Descent
%Perform linear regression using gradient descent and normal equation for cost optimization
%Split data into training and test sets
%Plot progress of GD over iterations
%Output prediction of y for a test set
%
% Requires the following functions: computeCostMulti.m,
% gradientDescentMulti.m
%
%Code based on ml-class.org (Ex.1)
%
%datasets used for development:
%http://archive.ics.uci.edu/ml/datasets/Housing
%http://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring
%http://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
%http://archive.ics.uci.edu/ml/datasets/Concrete+Slump+Test

%Input must contain feature columns followed by dependent variable column at end
data = load('simple_function_1.txt');

%Gradient Descent parameters
alpha = 0.01; num_iters = 1000;

%percentage of data to use for training
train_perc = .95;

%extract columns to use
X = data(:,1:end-1);
X_orig = X;
y = data(:,end);

%split into training and test sets:
test_rows = round(size(X,1)*(1-train_perc)); %number of rows to use in test set
X_test = X(1:test_rows,:); y_test = y(1:test_rows,:);%this is the test set
X = X(test_rows+1:end,:); y = y(test_rows+1:end,:);%this is the training set


%Use training set to get regression parameters:

%Compute mean and standard deviation, normalize X
mu = mean(X); sigma = std(X);
X = (X-repmat(mu,[size(X,1) 1]))./repmat(sigma,[size(X,1) 1]);
X_test = (X_test-repmat(mu,[size(X_test,1) 1]))./repmat(sigma,[size(X_test,1) 1]);

%Add intercept term to X
X = [ones(size(X,1), 1) X];
X_test = [ones(size(X_test,1), 1) X_test];

% %using normal equation
% theta = zeros(size(X,2), 1);
% theta = pinv(X' * X) * X' * y;

% Init Theta and Run Gradient Descent 
theta = zeros(size(X,2), 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure; plot(1:numel(J_history), J_history);
xlabel('Number of iterations');ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);

% Display cost function start and end values
fprintf('Cost function start: %g\n',J_history(1));
fprintf('Cost function end: %g\n',J_history(end));

%use test set for prediction - using mean & stdev of entire set
y_hat = X_test*theta;

%Compute training set error
J_train = computeCostMulti(X, y, theta);

%Compute test set error
J_test = computeCostMulti(X_test,y_test,theta);

%print test set variable - actual and prediction
fprintf('\nLinear fit:\n\n')
fprintf('\ty\t\ty_hat\n');
disp([y_test,y_hat]);
fprintf('\nTraining set error: %g\n',J_train);
fprintf('\nTest set error: %g\n',J_test);

%EOF





