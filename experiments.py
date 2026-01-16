import numpy as np
from linear_regression import LinearRegression

def generate_data(n_samples=100):
    X = np.random.rand(n_samples, 1)
    y = 3 * X.squeeze() + 2 + np.random.randn(n_samples) * 0.1
    return X, y

def run_experiment():
    x,y = generate_data()

    model = LinearRegression()
    model.fit(x,y)

    predictions = model.predict(x)
    print("First 5 predictions:", predictions[:5])

if __name__ == "__main__":
    run_experiment()
