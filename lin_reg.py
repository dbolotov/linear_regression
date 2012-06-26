#!/opt/EPD/bin/python
#Filename: lin_reg.py
#perform simple linear regression, compute training and test error, predict on test set
#code ported from linear_regression_script.m

#system python shebang: /usr/bin/python

import numpy
from numpy import mat, c_

#Input mut contain feature columns followed by dependent variable column at end
data = numpy.loadtxt('simple_function_1.txt', delimiter=',')

#gradient descent parameters
alpha = 0.01
num_iters = 1000.

#percentage of data to use for training
train_perc = 0.95

#separate input file into independent and dependent arrays
X = mat(data[:,:2]) #import as matrix
###X = data[:,:2] #import as array

y = mat(data[:,2])

#split into training and test sets
test_rows = int(round(X.shape[0] * (1 - train_perc))) #no. of rows in test set
X_test = X[:test_rows,:] #test set
y_test = y[:test_rows] #test set

X = X[test_rows:,:] #train set
y = y[test_rows:] #train setl


#Use training set to learn regression parameters

#Compute mean and standard deviation
mu = X.mean(axis=0)
sigma = X.std(axis=0)

#normalize sets
X = (X - mu)/sigma 
X_test = (X_test - mu)/sigma

#Add intercept term to sets
X = c_[numpy.ones(X.shape[0]), X]
X_test = c_[numpy.ones(X_test.shape[0]), X_test]

#Initialize theta and run gradient descent
theta = mat(numpy.zeros((X.shape[1],1)))

#Plot the GD convergence 

#Display GD result

#Display cost function intial and end values

#Use test set for prediction

#Compute training set error

#Compute test set error

#Print test set variable - actual and prediction

#EOF




































