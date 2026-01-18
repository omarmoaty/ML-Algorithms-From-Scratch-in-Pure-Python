import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
from logistic_regression import Logestic_Regression
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
    x=np.random.randn(n_samples,2)
    y=(x[:,0]+x[:,1]>0).astype(int)
    return x,y
def run_logistic_experiment():
    x,y = generate_classification_data()

    model = Logestic_Regression()
    model.fit(x,y)

    predictions = model.predict(x)
    accuracy = np.mean(predictions==y)

    print('training accuracy = ',accuracy)
if __name__ == "__main__":
    #run_linear_experiment()
    run_logistic_experiment()



