import numpy as np
from collections import Counter
from DT import DT

def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    ids = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[ids], y[ids]

class RF:
    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        self.trees = []

        for _ in range(self.n_trees):
            tree = DT(min_samples_split=self.min_samples_split, max_depth=self.max_depth, n_feats=self.n_feats)
            
            X_sample, y_sample = bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.n_trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        return np.array([self._most_common_label(tree_preds) for tree_pred in tree_preds])
    
    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]