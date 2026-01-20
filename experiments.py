import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
from logistic_regression import Logestic_Regression
from knn import KNN

#experiment on linear Regression
def generate_data(n_samples=100):
    X = np.random.rand(n_samples, 1)
    y = 3 * X.squeeze() + 2 + np.random.randn(n_samples) * 0.1
    return X, y

def run_linear_experiment():
    x,y = generate_data()

    model = LinearRegression()
    model.fit(x,y)

    print("Final loss:", model.loss_history[-1])

    predictions = model.predict(x)
    print("First 5 predictions:", predictions[:5])

    plt.plot(model.loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Linear Regression Convergence")
    plt.show()

#experiment on Logistic Regression
def generate_classification_data(n_samples=2000):
    X=np.random.randn(n_samples,2)
    noise = np.random.randn(n_samples) * 0.2
    y = ((X[:, 0] + X[:, 1] + noise) > 0).astype(int)
    return X,y
def run_logistic_experiment():
    x,y = generate_classification_data()

    model = Logestic_Regression(learning_rate=.01,epochs=50)
    model.fit(x,y)

    predictions = model.predict(x)
    accuracy = np.mean(predictions==y)

    plt.plot(model.loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("Logistic Regression Training Loss")
    plt.show()

    print('training accuracy = ',accuracy)

#experiment on KNN algorithm
def run_knn_experiment():
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 3],
        [6, 5],
        [7, 7],
        [8, 6]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = KNN(k=3)
    model.fit(X, y)

    predictions = model.predict(X)
    print("Predictions:", predictions)

if __name__ == "__main__":
    #run_linear_experiment()
    #run_logistic_experiment()
    run_knn_experiment()