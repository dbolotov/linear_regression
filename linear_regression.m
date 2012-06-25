function [theta, y_h, J_train, J_test] = linear_regression (dataset, alpha, num_iters, train_frac, alg_flag, verb_flag)
%LINEAR_REGRESSION Performs linear regression using several techniques for
%cost optimization. Plots progress of GD over iterations, output fitting
%parameters and prediction of y for a test set.
%
% [theta, y_h] = LINEAR_REGRESSION (dataset, alpha, num_iters, alg_flag, verb_flag)
% example call: [theta, y_h] = linear_regression ('data_file.txt', 0.01, 1000, 0.9, 0, 1);
%
% Input:
% dataset: .txt format, without header
% alpha: learning rate
% num_iters: number of iterations for gradient descent
% train_frac: fraction of dataset to use for training (0 to 1)
% alg_flag: specify optimization method (0 = gradient descent, 1 = normal equation)
% verb_flag: plot GD convergence and display regression results (0 = don't display, 1 = do display)
%
% Output:
% theta: learned regression parameters
% y_h: predicted output using test set
% J_train: training set error
% J_test: test set error
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

%to do
%add error metric
%add randomization to test and training sets

%to do when functionize:
%ask if file contains header
%ask if want to randomize selection of sets

%Input must contain feature columns followed by dependent variable column at end
data = load(dataset);

%extract columns to use
X = data(:,1:end-1);
y = data(:,end);

%split into training and test sets:
test_rows = round(size(X,1)*(1-train_frac)); %number of rows to use in test set
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

%Initialize theta
theta = zeros(size(X,2), 1);

if alg_flag == 0 %use Gradient Descent
    [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
elseif alg_flag == 1 % %usine normal equation
    theta = pinv(X' * X) * X' * y;
end 

%use test set for prediction - using mean & stdev of entire set
%y_h = [ones(size(X_test,1),1) (X_test-repmat(mu,[size(X_test,1) 1]))./repmat(sigma,[size(X_test,1) 1])]*theta;
y_h = X_test*theta;

%Compute training set error
J_train = computeCostMulti(X, y, theta);

%Compute test set error
J_test = computeCostMulti(X_test,y_test,theta);

if verb_flag == 1% % % Display metrics, GD tracking, etc.
    
    if alg_flag == 0 %cost function convergence info for gradient descent
        % Plot the convergence graph
        figure; plot(1:numel(J_history), J_history);
        xlabel('Number of iterations');ylabel('Cost J');

        % Display cost function start and end values
        fprintf('\nCost function start: %g\n',J_history(1));
        fprintf('Cost function end: %g\n',J_history(end));
    end
    
    %Display training and test set errors:
    fprintf('\nTraining set error: %g\n',J_train);
    fprintf('\nTest set error: %g\n',J_test);
    
	%print test set variable - actual and prediction
	fprintf('\nLinear fit (y, y_h):\n\n');
	disp([y_test,y_h]);
end
end

%EOF





