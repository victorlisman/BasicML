import numpy as np

class Per:
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs

        self.weights = None
        self.biases = np.random.rand(1).item()

    def fit(self, X, y):
        num_samples, num_features = X.shape

        self.weights = np.random.rand(num_features)

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.epochs):
            for i, x_i in enumerate(X):
                linear = np.dot(x_i, self.weights) + self.biases
                y_pred = self.unit_step(linear)

                self.weights += self.lr * (y_[i] - y_pred) * x_i
                self.bias += self.lr * (y_[i] - y_pred)


    def predict(self, X):
        linear = np.dot(X, self.weights) + self.biases
        return self.unit_step(linear)

    def unit_step(self, x):
        return np.where(x >= 0, 1, 0)