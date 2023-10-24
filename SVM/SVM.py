import numpy as np

class SVM:
    def __init__(self, lr, lambda_par, epochs):
        self.lr = lr
        self.lambda_par = lambda_par
        self.epochs = epochs
        
        self.w = None
        self.b = np.random.rand(1).item()

    def fit(self, X, y):
        num_samples, num_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.random.rand(num_features)

        for _ in range(self.epochs):
            for i, x_i in enumerate(X):
                if y_[i] * (np.dot(x_i, self.w) - self.b) >= 1:
                    self.w -= self.lr * (2 * self.lambda_par * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_par * self.w - np.dot(x_i, y_[i]))
                    self.b -= self.lr * y_[i]   


    def pred(self, X):
        return np.sign(np.dot(X, self.w) - self.b)