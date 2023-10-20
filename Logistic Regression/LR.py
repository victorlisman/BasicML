import numpy as np

# Same model as the Linear Regression one with the added sigmoid function to each predicted value

def sig(x):
    return 1 / (1 + np.exp(-x))

class LR:
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs

        self.weights = None
        self.biases = None

    def fit(self, X, y):
        num_samples, num_features = X.shape()

        self.weights = np.random.rand(num_features)
        self.biases = np.random.rand(1).item()

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.biases
            y_pred = sig(linear_model)

            dw = 1 / num_samples *(np.dot(X.T, (y_pred - y)))
            db = 1 / num_samples *(np.sum(y_pred - y))

            self.w -= dw * self.lr
            self.b -= db * self.lr

    def pred(self, X):
        return sig(np.dot(X, self.weights) + self.biases)

    


