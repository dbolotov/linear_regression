###Linear Regression Template

####Description
Linear regression algorithm implemented in MATLAB and python.

Regression parameters are learned using gradient descent or normal equation.
Allows any (reasonable) number of continuous features.
Splits input (csv file) into training and test sets.
Uses training set to learn parameters, and computes error on both training and test sets.
Plots cost function convergence if using gradient descent (set option in script).

Code based on Ex.1 of [ml-class.org](http://ml-class.org).

####Datasets used in development and testing
[Housing Dataset](http://archive.ics.uci.edu/ml/datasets/Housing)
[Parkinsons Telemonitoring Data Set](http://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring)

####Files
linear\_regression\_script.m: development script to perform regression.

linear\_regression.m: above script, in function form. Allows specification of learning parameters.

lin\_reg.py: python implementation of regression script.

