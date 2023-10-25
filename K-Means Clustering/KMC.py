import numpy as np

class KMC:
    def __init__(self, k, max_iters):
        self.k = k
        self.max_iters = max_iters

        self.clusters = [[] for _ in range(self.k)]
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        radnom_samples_is = np.random.choice(self.n_samples, self.k, replace=False)
        centroids_old = self.centroids
        self.centroids = [self.X[i] for i in radnom_samples_is]

        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(self.centroids)
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)

        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                labels[sample_i] = cluster_i

        return labels

    def _distance(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))
    
    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.k)]

        for i, sample in enumerate(self.X):
            centroid_i = self._closest_centroid(sample, centroids)
            clusters[centroid_i].append(i)

        return clusters
    
    def _closest_centroid(self, sample, centroids):
        distances = [self._distance(sample, point) for point in centroids]
        closest_i = np.argmin(distances)

        return closest_i
    
    def _get_centroids(self, clusters):
        centroids = np.random.rand((self.k, self.n_features))

        for cluster_i, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_i] = cluster_mean

        return centroids
    
    def _is_converged(self, centr_o, centr):
        distances = [self._distance(centr_o[i], centr[i]) for i in range(self.k)]
        return sum(distances) == 0