import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

def generate_data(n_samples=100):
    X = np.random.rand(n_samples, 1)
    y = 3 * X.squeeze() + 2 + np.random.randn(n_samples) * 0.1
    return X, y

def run_experiment():
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


if __name__ == "__main__":
    run_experiment()



