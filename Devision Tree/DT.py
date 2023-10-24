import numpy as np
from collections import Counter

def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)

    return -np.sum([p * np.log2(p) for p in ps if p > 0])

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DT:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        feat_i = np.random.choice(n_features, self.n_feats, replace=False)

        best_feature, best_thresh = self._best_criteria(X, y, feat_i)
        left_is, right_is = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_is, :], y[left_is], depth + 1)
        right = self._grow_tree(X[right_is, :], y[right_is], depth + 1)

        return Node(best_feature, best_thresh, left, right)
        
    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def _best_criteria(self, X, y, feat_is):
        best_gain = -1
        split_i, split_thresh = None, None

        for feat_i in feat_is:
            X_col = X[:, feat_i]
            thresholds = np.unique(X_col)

            for threshold in thresholds:
                gain = self._information_gain(y, X_col, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_i = feat_i
                    split_thresh = threshold

        return split_i, split_thresh
    
    def _information_gain(self, y, X_col, split_thresh):
        parent_ent = entropy(y)

        left_is, right_is = self._split(X_col, split_thresh)

        if len(left_is) == 0 or len(right_is) == 0:
            return 0
        
        n = len(y)
        n_l, n_r = len(left_is), len(right_is)
        e_l, e_r = entropy(y[left_is]), entropy(y[right_is])

        child_ent = (n_l / n) * e_l + (n_r / n) * e_r

        return parent_ent - child_ent

    def _split(self, X_col, split_thresh):
        left_is = np.argwhere(X_col <= split_thresh).flatten()
        right_is = np.argwhere(X_col > split_thresh).flatten()

        return left_is, right_is    

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        
        return self._traverse_tree(x, node.right)