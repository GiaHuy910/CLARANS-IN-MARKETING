import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


class CLARANS:
    """CLARANS (Clustering Large Applications based on RANdomized Search) algorithm"""
    
    def __init__(self, data, n_clusters, num_local, max_neighbors):
        self.data = data
        self.n_clusters = n_clusters
        self.num_local = num_local
        self.max_neighbors = max_neighbors
        self.labels = None
        self.medoids = None
        self.sihouette_score_ = None
        self.Inertia_ = None

    def _euclid_dist(self, X, medoids):
        return np.linalg.norm(X[:, np.newaxis] - medoids, axis=2)

    def _compute_total_cost(self, X, medoids):
        distances = self._euclid_dist(X, medoids)
        return np.sum(np.min(distances, axis=1))

    def fit(self):
        X = self.data.values
        best_cost = float('inf')

        for _ in range(self.num_local):
            medoids_idx = np.random.choice(len(X), self.n_clusters, replace=False)
            medoids = X[medoids_idx]

            current_cost = self._compute_total_cost(X, medoids)
            neighbor = 0
            while neighbor < self.max_neighbors:
                medoids_random_pos = np.random.randint(self.n_clusters)
                candidate_idx = np.random.randint(len(X))
                while candidate_idx in medoids_idx:
                    candidate_idx = np.random.randint(len(X))

                new_medoids_idx = medoids_idx.copy()
                new_medoids_idx[medoids_random_pos] = candidate_idx
                new_medoids = X[new_medoids_idx]

                new_cost = self._compute_total_cost(X, new_medoids)
                if current_cost > new_cost:
                    medoids = new_medoids
                    medoids_idx = new_medoids_idx
                    current_cost = new_cost
                    neighbor = 0
                else:
                    neighbor += 1
            if current_cost < best_cost:
                best_cost = current_cost
                self.medoids = medoids
        distances = self._euclid_dist(X, self.medoids)
        self.labels = np.argmin(distances, axis=1)
        self.Inertia_ = best_cost

    def get_medoids(self):
        return self.medoids

    def get_labels(self):
        return self.labels

    def get_Inertia(self):
        return self.Inertia_


class KMEANS:
    """K-Means clustering algorithm"""
    
    def __init__(self, data, n_clusters, max_iters=300):
        self.data = data
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.labels = None
        self.centroids = None
        self.Inertia = None

    def euclidean_distance(self, point1, point2):
        return np.linalg.norm(point1 - point2, axis=2)

    def fit(self):
        X = self.data.values
        random_idx = np.random.choice(len(X), size=self.n_clusters, replace=False)
        centroids = X[random_idx]

        for _ in range(self.max_iters):
            distances = self.euclidean_distance(X[:, np.newaxis], centroids)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        self.centroids = centroids
        self.labels = labels
        self.Inertia = np.sum(np.linalg.norm(X - centroids[labels], axis=1) ** 2)

    def get_centroids(self):
        return self.centroids

    def get_labels(self):
        return self.labels

    def get_Inertia(self):
        return self.Inertia


class PAM:
    """PAM (Partitioning Around Medoids) clustering algorithm"""
    
    def __init__(self, data, n_clusters=5):
        self.n_clusters = n_clusters
        self.data = data
        self.medoids_ = None
        self.labels_ = None
        self.Inertia = None

    def _euclid_dist(self, a, b):
        return np.linalg.norm(a - b, axis=2)

    def assign_clusters(self, points, medoids):
        distances = self._euclid_dist(points[:, np.newaxis], medoids)
        labels = np.argmin(distances, axis=1)
        return labels

    def fit(self):
        X = self.data.values
        random_idx = np.random.choice(len(X), self.n_clusters, replace=False)
        medoids = X[random_idx]

        # First cluster assignment
        labels = self.assign_clusters(X, medoids)
        new_medoids = []

        for i in range(self.n_clusters):
            clusters_points = X[labels == i]

            if len(clusters_points) == 0:
                new_medoids.append(medoids[i])
                continue
            costs = []

            for candidate in clusters_points:
                cost = np.sum(np.linalg.norm(clusters_points - candidate, axis=1))
                costs.append(cost)
            best_medoids = clusters_points[np.argmin(costs)]
            new_medoids.append(best_medoids)
        new_medoids = np.array(new_medoids)

        self.medoids_ = new_medoids
        self.labels_ = labels
        self.Inertia = np.sum(np.linalg.norm(X - self.medoids_[self.labels_], axis=1))

    def get_medoids(self):
        return self.medoids_

    def get_labels(self):
        return self.labels_

    def get_Inertia(self):
        return self.Inertia
