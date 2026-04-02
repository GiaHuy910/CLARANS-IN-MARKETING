import numpy as np
class KMEANS:
    def __init__(self,data,n_clusters,max_iters=300):
        self.data = data
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.labels=None
    def euclidean_distance(self,point1,point2):
        return np.linalg.norm(point1-point2,axis=2)
    def fit(self):
        X=self.data.values
        random_idx=np.random.choice(len(X),size=self.n_clusters,replace=False)
        centroids=X[random_idx]

        for _ in range(self.max_iters):
            distances=self.euclidean_distance(X[:,np.newaxis],centroids)

            labels=np.argmin(distances,axis=1)
            new_centroids=np.array([X[labels==i].mean(axis=0) for i in range(self.n_clusters)])
            if np.allclose(centroids,new_centroids):
                break
            self.centroids=new_centroids
        self.centroids=centroids
        self.labels=labels
    def get_centroids(self):
        return self.centroids
    def get_labels(self):
        return self.labels