#this file implements Linear Regression from scratch using gradient descent and mean-squared-error
import random
class LinearRegression:
	def __init__(self, learning_rate = 0.01, epochs = 1000):
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.weights = None
		self.bias = None
	
