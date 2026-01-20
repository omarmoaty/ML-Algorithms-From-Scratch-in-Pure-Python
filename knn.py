import numpy as np

class KNN:
    def __init__(self, k = 3):
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def euclidian_distance(self, x1, x2):
        return np.sqrt(np.sum( (x1-x2)**2))

    def predict(self, X):
        predictions = []

        for x in X:
            distances = [ self.euclidian_distance(x,x_train) for x_train in self.x_train]

            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]

            most_common = np.bincount(k_labels).argmax()
            predictions.append(most_common)

        return np.array(predictions)
    
