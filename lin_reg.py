#!/opt/EPD/bin/python
#Filename: lin_reg.py
#perform simple linear regression, compute training and test error, predict on test set
#code ported from linear_regression_script.m


import numpy
from numpy import mat, c_

#Define functions
def gradientDescentMulti(X, y, theta, alpha, num_iters):
	J_history = 0.
	return theta,J_history

def computeCostMulti(X,y,theta):
	J = 0.
	return J

#Input must contain feature columns followed by dependent variable column at end
data = numpy.loadtxt('simple_function_1.txt', delimiter=',')

#gradient descent parameters
alpha = 0.01
num_iters = 1000.

#percentage of data to use for training
train_perc = 0.95

#separate input file into independent and dependent arrays
X = mat(data[:,:2]) #import as matrix
###X = data[:,:2] #import as array

y = numpy.transpose(mat(data[:,2]))

#split into training and test sets
test_rows = int(round(X.shape[0] * (1 - train_perc))) #no. of rows in test set
X_test = X[:test_rows,:] #test set
y_test = y[:test_rows] #test set

X = X[test_rows:,:] #train set
y = y[test_rows:] #train set


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

theta,J_history = gradientDescentMulti(X,y,theta,alpha,num_iters)

#Plot the GD convergence 

#Display GD result
print "Theta computed from gradient descent:\n",theta

#Display cost function intial and end values
print "Cost function start:\n"
print "Cost function end:\n"

#Use test set for prediction
y_hat = X_test*theta

#Compute training set error
J_train = computeCostMulti(X, y, theta)

#Compute test set error
J_test = computeCostMulti(X_test,y_test,theta)

#Print test set variable - actual and prediction
print "Linear fit: actual vs. prediction\n"



#EOF




