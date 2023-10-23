import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

class NB:
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self._classes = np.unique(y)    
        num_classes = len(self._classes)

        self._mean = np.zeros((num_classes, num_features), dtype=np.float64)
        self._var = np.zeros((num_classes, num_features), dtype=np.float64)
        self._priors = np.zeros(num_classes, dtype=np.float64)

        for c in self._classes:
            X_c = X[c == y]
            self._mean[c, :] = X_c.mean(axis=0)
            self._var[c, :] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(num_samples)

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        posteriors = []

        for i, c in enumerate(self._classes):
            prior = np.log(self._priors[i])
            class_conditional = np.sum(np.log(self._pdf(i, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_i, x):
        return 1 / np.sqrt(2 * np.pi * self._var[class_i]) * np.exp(-(x - self._mean[class_i]) ** 2 / (2 * self._var[class_i]))

def accuracy(y_true, y_pred):
    return np.sum(y_pred == y_true) / len(y_true)
