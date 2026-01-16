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
		self.loss_history = []
		n_samples, n_features = x.shape
		self.weights = np.zeros(n_features)
		self.bias = 0.0

		for _ in range(self.epochs):
			y_pred = self.predict(x)

			dw = (1/n_samples) * np.dot(x.T, (y_pred - y))
			db = (1/n_samples) * np.sum(y_pred - y)

			self.weights -= self.learning_rate * dw
			self.bias -= self.learning_rate * db

			loss = self._mse(y,y_pred)
			self.loss_history.append(loss)
	
	def predict(self, x):
		return np.dot(x,self.weights) + self.bias 
	
	def _mse(self,y_true, y_pred):
		return np.mean((y_true-y_pred)**2)
	
