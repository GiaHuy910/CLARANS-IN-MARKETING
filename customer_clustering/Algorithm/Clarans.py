import random
import numpy as np
class CLARANS:
    def __init__(self,data,n_clusters,num_local,max_neighbors):
        self.data=np.array(data)
        self.n_clusters=n_clusters
        self.num_local=num_local
        self.max_neighbors=max_neighbors

        self.medoids_: list[int] = []
        self.clusters_: list[list[int]] = []
        self.cost_: float = float("inf")
        self.sihouette_score_: float = -1.0
    def _euclid_dist(self,a,b):
        return  np.linalg.norm(a - b)
    def _compute_total_cost(self,medoids):
        cost = 0
        clusters=[[] for _ in range(len(medoids))]
        for i,point in enumerate(self.data):
            distances=np.array([self._euclid_dist(point,self.data[m]) for m in medoids])
            cluster_id=int(np.argmin(distances))
            clusters[cluster_id].append(i)
            cost += distances[cluster_id]
        return cost,clusters
    def fit(self):
        n=len(self.data)
        best_medoids=None
        best_clusters=None
        best_cost=float('inf')
        for _ in range(self.num_local):
            medoids=random.sample(range(n),self.n_clusters)
            current_cost,current_clusters=self._compute_total_cost(medoids)
            neighbor=0
            while neighbor<self.max_neighbors:
                medoids_idx=random.choice(range(self.n_clusters))
                non_medoids=list(set(range(n))-set(medoids))
                new_medoid=random.choice(non_medoids)

                new_medoids=medoids.copy()
                new_medoids[medoids_idx]=new_medoid

                new_cost,new_clusters=self._compute_total_cost(new_medoids)
                if new_cost<current_cost:
                    medoids=new_medoids
                    current_cost=new_cost
                    current_clusters=new_clusters
                    neighbor=0
                else:
                    neighbor+=1
            if current_cost<best_cost:
                best_cost=current_cost
                best_medoids=medoids
                best_clusters=current_clusters
        self.medoids=best_medoids
        self.clusters=best_clusters
        self.cost=best_cost
    def get_clusters(self):
        return self.clusters
    def get_medoids(self):
        return self.medoids
    def get_labels(self):
        clusters = self.get_clusters()
        labels = [0] * len(self.data)
        for cluster_id, cluster in enumerate(clusters):
            for index in cluster:
                labels[index] = cluster_id
        return labels

