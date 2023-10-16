import numpy as np
from collections import Counter

# KNN clasification 

# Compute euclidean distance between two points
def distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

class KNN:
    # Assign number of neighbors
    def __init__(self, k=3):
        self.k = k

    # Fit the data to the KNN model X_train = features, y_train = labels
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Return the predicted
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    # Helper function
    def _predict(self, x):
        # Compute distance between sample and features
        distances = [distance(x, x_train) for x_train in self.X_train]

        # Get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Most common class label
        most_common = Counter(k_nearest_labels).most_common(1)

        return most_common[0][0]
    
if __name__ == "__main__":
    # Imports
    from matplotlib.colors import ListedColormap
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    k = 3
    clf = KNN(k=k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("KNN classification accuracy", accuracy(y_test, predictions))