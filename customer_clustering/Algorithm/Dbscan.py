import numpy as np
from scipy.spatial import cKDTree

class DBSCAN:
    def __init__(self, data, eps=0.5, min_samples=5):
        self.data = np.array(data)
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.clusters_ = []
        self.n_clusters_ = 0
        self.noise_points_ = []
        self.silhouette_score_ = -1.0
        self.noise_ratio_ = 0.0        
        self.kdtree = None
    
    def _euclidean_dist(self, a, b):
        return np.linalg.norm(a - b)
    
    def _get_neighbors(self, point_idx):
        neighbors = self.kdtree.query_ball_point(self.data[point_idx], self.eps)
        return neighbors
    
    def _expand_cluster(self, point_idx, cluster_id, visited, neighbors):
        self.labels_[point_idx] = cluster_id
        
        queue = list(neighbors)
        while queue:
            neighbor_idx = queue.pop(0)
            
            if neighbor_idx not in visited:
                visited.add(neighbor_idx)
                neighbor_neighbors = self._get_neighbors(neighbor_idx)
                
                # Nếu là core point, tiếp tục mở rộng cụm
                if len(neighbor_neighbors) >= self.min_samples:
                    queue.extend(neighbor_neighbors)
            
            # Gán vào cụm nếu chưa được gán
            if self.labels_[neighbor_idx] == -1:
                self.labels_[neighbor_idx] = cluster_id
    
    def fit(self):
        n_samples = len(self.data)
        
        # Xây dựng KD-tree
        self.kdtree = cKDTree(self.data)
        
        # Khởi tạo tất cả điểm là noise (-1)
        self.labels_ = np.array([-1] * n_samples)
        
        visited = set()
        cluster_id = 0
        
        for point_idx in range(n_samples):
            if point_idx in visited:
                continue
            
            visited.add(point_idx)
            neighbors = self._get_neighbors(point_idx)
            
            # Nếu không phải core point, giữ là noise
            if len(neighbors) < self.min_samples:
                continue
            
            # Mở rộng cụm từ core point này
            self._expand_cluster(point_idx, cluster_id, visited, neighbors)
            cluster_id += 1
        
        # Tính toán các metrics
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        self._compute_clusters()
        self._compute_noise_points()
        self._compute_silhouette_score()
    
    def _compute_clusters(self):
        self.clusters_ = [[] for _ in range(self.n_clusters_)]
        for i, label in enumerate(self.labels_):
            if label != -1:
                self.clusters_[label].append(i)
    
    def _compute_noise_points(self):
        self.noise_points_ = [i for i, label in enumerate(self.labels_) if label == -1]
        total_points = len(self.data)
        self.noise_ratio_ = len(self.noise_points_) / total_points if total_points > 0 else 0
    
    def _compute_silhouette_score(self):
        if self.n_clusters_ < 2:
            self.silhouette_score_ = -1.0
            return
        
        # Chỉ tính với điểm được gán cụm (không phải noise)
        clustered_indices = np.where(self.labels_ != -1)[0]
        clustered_labels = self.labels_[clustered_indices]
        
        if len(np.unique(clustered_labels)) < 2:
            self.silhouette_score_ = -1.0
            return
        
        try:
            silhouette_scores = self._compute_silhouette_samples(clustered_indices, clustered_labels)
            self.silhouette_score_ = np.mean(silhouette_scores)
        except:
            self.silhouette_score_ = -1.0
    
    def _compute_silhouette_samples(self, indices, labels):
        """Tính silhouette coefficient cho mỗi điểm"""
        silhouette_vals = []
        
        for idx, point_idx in enumerate(indices):
            point = self.data[point_idx]
            current_label = labels[idx]
            
            # Tính a(i): khoảng cách trung bình đến các điểm trong cụm hiện tại
            same_cluster_indices = indices[labels == current_label]
            if len(same_cluster_indices) > 1:
                distances_same = [self._euclidean_dist(point, self.data[i]) for i in same_cluster_indices if i != point_idx]
                a_i = np.mean(distances_same) if distances_same else 0
            else:
                a_i = 0
            
            # Tính b(i): khoảng cách trung bình nhỏ nhất đến các điểm trong cụm khác
            b_i = float('inf')
            unique_labels = np.unique(labels)
            for label in unique_labels:
                if label != current_label:
                    other_cluster_indices = indices[labels == label]
                    distances_other = [self._euclidean_dist(point, self.data[i]) for i in other_cluster_indices]
                    avg_distance = np.mean(distances_other)
                    b_i = min(b_i, avg_distance)
            
            # Tính silhouette coefficient
            if b_i == float('inf'):
                silhouette = 0
            else:
                max_dist = max(a_i, b_i)
                if max_dist == 0:
                    silhouette = 0
                else:
                    silhouette = (b_i - a_i) / max_dist
            
            silhouette_vals.append(silhouette)
        
        return np.array(silhouette_vals)
    
    def get_clusters(self):
        return self.clusters_
    
    def get_labels(self):
        return self.labels_
    
    def get_noise_points(self):
        return self.noise_points_
    
    def get_noise_ratio(self):
        return self.noise_ratio_
