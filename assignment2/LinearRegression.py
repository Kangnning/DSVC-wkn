import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

def LinearRegression(X, y, alpha=1, num_iters=100):
	X.insert(0, 'x0', 1)
	X_train = np.array(X)
	y_train = np.array(y).reshape(len(y), 1)
	X_train[:, 1] = (X_train[:, 1] - np.mean(X_train[:, 1])) / np.std(X_train[:, 1])
	X_train[:, 2] = (X_train[:, 2] - np.mean(X_train[:, 2])) / np.std(X_train[:, 2])
	X_train[:, 3] = (X_train[:, 3] - np.mean(X_train[:, 3])) / np.std(X_train[:, 3])
	theta = np.ones(X.shape[1]).reshape(X.shape[1], 1)
	theta = GradientDescent(X_train, y_train, theta, alpha, num_iters)
	print(theta)
	predict(X_train, y_train, theta)

def GradientDescent(X, y, theta, alpha, num_iters):
	while(num_iters):
		num_iters = num_iters - 1
		for i in np.arange(X.shape[1]):
			g = np.dot(X[:, i], (np.dot(X, theta) - y)) * alpha / len(y)
			theta[i] = theta[i] - g
			cost = Costfun(X, y, theta)
	return theta

def Costfun(X, y, theta):
	cost = np.sum((np.dot(X, theta) - y)**2) / (2 * len(y))
	return cost

def predict(X, y, theta):
	y_pre = np.dot(X, theta)
	print(y_pre)
	print(r2_score(y, y_pre))


data = pd.read_csv('E://DSVC-master/assignment2/homework/housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
LinearRegression(features, prices)