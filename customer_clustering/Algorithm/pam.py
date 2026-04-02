import numpy as np
class PAM:
    def __init__(self, n_clusters = 5, max_iter = 300):
        self.n_clusters        = n_clusters
        self.max_iter          = max_iter
        self.medoids_          = []
        self.clusters_         = []
        self.cost_             = float('inf')
        self.silhouette_score_ = -1.0
        self.data              = np.array([])

    def _euclid_dist(self, a, b):
        return np.linalg.norm(a - b)

    def _compute_total_cost(self, medoids):
        total = 0.0
        for point in self.data:
            total += min(self._euclid_dist(point, self.data[m]) for m in medoids)
        return total

    def _assign_clusters(self, medoids):
        clusters = [[] for _ in range(self.n_clusters)]
        for i, point in enumerate(self.data):
            distances  = [self._euclid_dist(point, self.data[m]) for m in medoids]
            cluster_id = int(np.argmin(distances))
            clusters[cluster_id].append(i)
        return clusters

    def fit(self, data):
        self.data = np.array(data.values)
        n         = len(self.data)
        medoids   = list(np.random.choice(n, self.n_clusters, replace = False))
        best_cost = self._compute_total_cost(medoids)

        for _ in range(self.max_iter):
            improved = False
            for i in range(self.n_clusters):
                for candidate in range(n):
                    if candidate in medoids:
                        continue
                    new_medoids    = medoids.copy()
                    new_medoids[i] = candidate
                    new_cost       = self._compute_total_cost(new_medoids)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        medoids   = new_medoids.copy()
                        improved  = True
            if not improved:
                break

        self.medoids_  = medoids
        self.clusters_ = self._assign_clusters(medoids)
        self.cost_     = best_cost
        return self

    def get_clusters(self):
        return self.clusters_

    def get_medoids(self):
        return self.medoids_

    def get_labels(self):
        labels = [0] * len(self.data)
        for cluster_id, cluster in enumerate(self.clusters_):
            for index in cluster:
                labels[index] = cluster_id
        return labels

