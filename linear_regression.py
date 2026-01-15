#this file implements Linear Regression from scratch using gradient descent and mean-squared-error
import random
import numpy as np
class LinearRegression:
	def __init__(self, learning_rate = 0.01, epochs = 1000):
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.weights = None
		self.bias = None
	
	def fit(self, x, y):
		n_features = x.shape[1]
		self.weights = np.zeros(n_features)
		self.bias = 0.0
	
	def predict(self, x):
		return np.dot(x,self.weights) + self.bias
