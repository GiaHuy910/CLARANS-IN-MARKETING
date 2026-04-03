import random
import numpy as np
class CLARANS:
    def __init__(self,data,n_clusters,num_local,max_neighbors):
        self.data=data
        self.n_clusters=n_clusters
        self.num_local=num_local
        self.max_neighbors=max_neighbors
        self.labels=None
        self.medoids=None
        self.sihouette_score_=None

    def _euclid_dist(self,X,medoids):
        return  np.linalg.norm(X[:, np.newaxis] -medoids,axis=2)
    def _compute_total_cost(self,X,medoids):
        distances=self._euclid_dist(X,medoids)
        return np.sum(np.min(distances,axis=1))
    def fit(self):
        X=self.data.values
        best_cost=float('inf')

        for _ in range(self.num_local):
            medoids_idx=np.random.choice(len(X),self.n_clusters,replace=False)
            medoids=X[medoids_idx]

            current_cost=self._compute_total_cost(X,medoids)
            neighbor=0
            while neighbor<self.max_neighbors:
                medoids_random_pos=np.random.randint(self.n_clusters)
                candidate_idx=np.random.randint(len(X))

                new_medoids=medoids.copy()
                new_medoids[medoids_random_pos]=X[candidate_idx]
                new_cost=self._compute_total_cost(X,new_medoids)
                if current_cost<new_cost:
                    medoids=new_medoids
                    current_cost=new_cost
                    neighbor=0
                else:
                    neighbor+=1
            if current_cost<best_cost:
                best_cost=current_cost
                self.medoids=medoids
        distances=self._euclid_dist(X,self.medoids)
        self.labels=np.argmin(distances,axis=1)
        self.Inertia_=best_cost
    def get_medoids(self):
        return self.medoids
    def get_labels(self):
        return self.labels
    def get_Inertia(self):
        return self.Inertia_
