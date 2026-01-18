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
    
    def _binary_cross_entropy_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred,epsilon,1-epsilon)
        return -np.mean( y_true * np.log(y_pred) + (1-y_true) * np.log(1- y_pred) )
    
    def fit(self, x, y):
        n_samples,n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.epochs):
            y_pred = self.predict_probability(x)

            dw = (1 / n_samples) * np.dot(x.T,(y_pred - y))
            db = (1 / n_samples) * np.sum( y_pred-y )

            self.weights = -self.learning_rate * dw
            self.bias = - self.learning_rate * db

            loss = self._binary_cross_entropy_loss(y, y_pred)
            self.loss_history.append(loss)