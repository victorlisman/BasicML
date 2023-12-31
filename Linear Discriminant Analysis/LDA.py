import numpy as np

class LDA:
    def __init__(self, n_comps):
        self.n_comps = n_comps
        self.lin_dis = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        mean_overall = np.mean(X, axis=0)
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))

        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            
            S_W += (X_c - mean_c).T.dot(X_c - mean_c)

            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            S_B += n_c * (mean_diff).dot(mean_diff.T)

        A = np.linalg.inv(S_W).dot(S_B)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvectors = eigenvectors.T
        ies = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[ies]
        eigenvectors = eigenvectors[ies]
        self.lin_dis = eigenvectors[0:self.n_comps]

    def transform(self, X):
        return np.dot(X, self.lin_dis.T)