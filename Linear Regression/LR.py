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

            dw = (1 / num_samples) *(np.dot(X.T, (y_pred - y)))
            db = (1 / num_samples) *(np.sum(y_pred - y))

            self.w = self.w - (dw * self.lr)
            self.b -= db * self.lr

    # Return the prediction for a new X matrix
    def pred(self, X):
        y_pred = np.dot(X, self.w) + self.b

        return y_pred
        

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


X_train = [1, 2, 3, 4, 5]
y_train = [2, 4, 6, 8, 10]

X_train = np.array(X_train, dtype=np.float64).reshape(-1, 1)
y_train = np.array(y_train, dtype=np.float64).reshape(-1, 1)
regr = LR(lr=0.001, epochs=10000)
regr.fit(X_train, y_train)
X_new = np.array([20, 40, 50, 60, 70]).reshape(-1, 1)
y_new = np.array([40, 80, 100, 120, 140]).reshape(-1, 1)
y_pred = regr.pred(X_new)
print(MSE(y_new, y_pred))
print(y_pred[:, 0])