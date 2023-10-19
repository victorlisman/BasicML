import numpy as np

class LR:
    # Assign the learning rate and number of epochs for this model and initialize the weights and biases
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs

        self.w = None
        self.b = None

    def fit(self, X, y):
        # Extract the number of samples(number of vectors) and features(number of elements in the matrix)
        num_samples, num_features = X.shape

        # Set the weights to a vector of random numbers with the same size of elements as X, and set a scalar value for the bias
        self.w = np.random.rand(num_features)
        self.b = np.random.rand(1).item()

        # Train the model and calculate the gradient for the weights and biases then adjust them using the learning rate
        for _ in range(self.epochs):
            y_pred = np.dot(X, self.w) + self.b

            dw = 1 / num_samples *(np.dot(X.T, (y_pred - y)))
            db = 1 / num_samples *(np.sum(y_pred - y))

            self.w -= dw * self.lr
            self.b -= db * self.lr

    # Return the prediction for a new X matrix
    def pred(self, X):
        y_pred = np.dot(X, self.w) + self.b

        return y_pred
        
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

regr = LR(lr=0.001, epochs=10000)
regr.fit(X_train, y_train)
predicted = regr.pred(X_test)

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

mse = MSE(y_test, predicted)
print(mse)

plt.scatter(X_test, y_test)
plt.plot(X_test, predicted, color='red')

plt.show()

