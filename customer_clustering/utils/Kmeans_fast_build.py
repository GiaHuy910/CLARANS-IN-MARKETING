from Algorithm import KMEANS
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
def Kmeans_silhouette_analysis(data, k_range):
    silhouette_scores = []
    for k in k_range:
        model = KMEANS(data,n_clusters=k)
        model.fit()
        #kmeans get labels

        silhouette_scores.append(silhouette_score(data, model.get_labels()))
        print(f'k={k}, Silhouette Score: {silhouette_score(data, model.get_labels()):.4f}')

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for CLARANS')
    plt.show()