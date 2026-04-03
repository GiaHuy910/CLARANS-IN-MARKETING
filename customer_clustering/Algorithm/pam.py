import numpy as np
class PAM:
    def __init__(self,data,n_clusters = 5):
        self.n_clusters        = n_clusters
        self.data              = data
    def _euclid_dist(self, a, b):
        return np.linalg.norm(a - b,axis=2)
    def assign_clusters(self,points, medoids):
        distances=self._euclid_dist(points[:,np.newaxis],medoids)
        labels=np.argmin(distances,axis=1)
        return labels
    def fit(self):
        X=self.data.values
        random_idx=np.random.choice(len(X),self.n_clusters,replace=False)
        medoids=X[random_idx]

        #lần gán nhãn đầu tiên
        labels=self.assign_clusters(X,medoids)
        new_medoids=[]

        for i in range(self.n_clusters):
            clusters_points=X[labels==i]

            if len(clusters_points)==0:
                new_medoids.append(medoids[i])
                continue
            costs=[]

            for candidate in clusters_points:
                cost=np.sum(np.linalg.norm(clusters_points-candidate,axis=1))
                costs.append(cost)
            best_medoids=clusters_points[np.argmin(costs)]
            new_medoids.append(best_medoids)
        new_medoids=np.array(new_medoids)

        self.medoids_=new_medoids
        self.labels_=labels
        self.Inertia=np.sum(np.linalg.norm(X - self.medoids_[self.labels_], axis=1))
    def get_medoids(self):
        return self.medoids_
    def get_labels(self):
        return self.labels_
    def get_Inertia(self):
        return self.Inertia