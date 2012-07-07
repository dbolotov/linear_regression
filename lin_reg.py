#!/usr/local/EPD/bin/python
#/usr/bin/env python
#Filename: lin_reg.py

#perform simple linear regression using batch gradient descent on training and test sets 

#Overview:
#take in a csv file (no header)
#split into a training and test set
#define regression parameters: learning rate, number of iterations
#perform regression to get fitting parameters
#plot gradient descent convergence
#compute training and test error, predict on test set

#code ported from linear_regression_script.m (based on Ex1 from ml-class.org)


import numpy
import matplotlib.pyplot as plt
from numpy import mat, c_,r_,array

#Define functions
def gradientDescentMulti(X, y, theta, alpha, num_iters):
	m = y.shape[0]
	J_history = numpy.zeros((num_iters,1))
	for iter in range(0,num_iters):
		#perform a single gradient step on the parameter vector theta
		theta = theta - alpha/m*(numpy.transpose(X)*(X*theta-y))
		#save the cost J in every iteration
		J_history[iter] = computeCostMulti(X,y,theta)
	return theta,J_history

def computeCostMulti(X,y,theta):
	m = y.shape[0]
	J = 0.5/m * (numpy.transpose(X*theta - y))*(X*theta-y)
	return J

#Linear regression script

#Input must contain feature columns followed by dependent variable column at end
data = numpy.loadtxt('simple_function_1.txt', delimiter=',')

#gradient descent parameters
alpha = 0.01 #learning rate
num_iters = 1000 #number of iterations for gradient descent

#percentage of data to use for training
train_perc = 0.95

#separate input file into independent and dependent arrays
X = mat(data[:,:2]) 
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
plt.plot(r_[:num_iters],J_history)
plt.xlabel('Iterations')
plt.ylabel('Cost J')
plt.title('Gradient Descent convergence plot')
plt.show()

#Display GD result
print "Theta computed from gradient descent:\n",theta

#Display cost function intial and end values
print "Cost function start: %g \n" % J_history[0]
print "Cost function end: %g \n" % J_history[-1]

#Use test set for prediction
y_hat = X_test*theta

#Compute training set error
J_train = computeCostMulti(X, y, theta)

#Compute test set error
J_test = computeCostMulti(X_test,y_test,theta)

#Print test set variable - actual and prediction
print "Linear fit: actual vs. prediction\n"
print c_[y_test,y_hat]
print "Training set error: %g \n" % J_train
print "Test set error: %g \n" % J_test
#EOF

