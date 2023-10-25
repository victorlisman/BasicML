import numpy as np

class PCA:
    def __init__(self, n_comps):
        self.n_comps = n_comps
        self.comps = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)

        X -= self.mean

        cov = np.cov(X.T)

        eigenvalues, eigenvectors = np.linalg.eig(cov)

        eigenvectors = eigenvectors.T

        ies = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[ies]
        eigenvectors = eigenvectors[ies]

        self.comps = eigenvectors[0:self.n_comps]

    def transform(self, X):
        X -= self.mean

        return np.dot(X, self.comps.T)