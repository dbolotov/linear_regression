Linear regression algorithm implemented in MATLAB and python.

Regression parameters are learned using gradient descent or normal equation.
Allows any (reasonable) number of features.
Splits input (csv file) into training and test sets.
Uses training set to learn parameters, and computes error on both training and test sets.
Plots cost function convergence if using gradient descent.

linear_regression_script.m: development script to perform regression.

linear_regression.m: above script, in function form. Allows specification of learning parameters.

lin_reg.py: python implementation of regression script.

Code based on Ex.1 of ml-class.org

Some datasets used for development:
ttp://archive.ics.uci.edu/ml/datasets/Housing
http://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring


