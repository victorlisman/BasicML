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
        self.w = np.zeros(num_features)
        self.b = 0

        # Train the model and calculate the gradient for the weights and biases then adjust them using the learning rate
        for _ in range(self.epochs):
            y_pred = np.dot(X, self.w) + self.b

            dw = (1 / num_samples) *(np.dot(X.T, (y_pred - y)))
            db = (1 / num_samples) *(np.sum(y_pred - y))

            self.w = self.w - (dw * self.lr)
            self.b = self.b - db * self.lr

    # Return the prediction for a new X matrix
    def predict(self, X):
        y_pred = np.dot(X, self.w) + self.b
        return y_pred
    
    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (numerator / denominator)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return self.mse(y, y_pred), self.r2_score(y, y_pred)
        

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

