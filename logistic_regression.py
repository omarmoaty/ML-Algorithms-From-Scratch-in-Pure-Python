import numpy as np

class Logestic_Regression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def predict_probability(self, x):
        linear_output = np.dot(x, self.weights) + self.bias
        return self._sigmoid(linear_output)