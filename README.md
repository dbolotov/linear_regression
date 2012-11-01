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

####Files
MATLAB regression script: linear\_regression\_script.m

Functional form of script (allows specification of parameters in function call): linear\_regression.m

Python implementation of regression script: lin\_reg.py

MATLAB functions used by main script and function: computeCostMulti.m, gradientDescentMulti.m

