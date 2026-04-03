from Algorithm import PAM
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score
def PAM_silhouette_analysis(data, k_range,print_scores=False):
    silhouette_scores = []
    for k in k_range:
        model = PAM(data,n_clusters=k)
        model.fit()
        #PAM get labels

        silhouette_scores.append(silhouette_score(data, model.get_labels()))
        if print_scores:
            print(f'k={k}, Silhouette Score: {silhouette_score(data, model.get_labels()):.4f}')

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for PAM')
    plt.show()
def PAM_davies_bouldin_analysis(data, k_range,print_scores=False):
    davies_bouldin_scores = []
    for k in k_range:
        model = PAM(data,n_clusters=k)
        model.fit()
        davies_bouldin_scores.append(davies_bouldin_score(data, model.get_labels()))
        if print_scores:
            print(f'k={k}, Davies-Bouldin Score: {davies_bouldin_score(data, model.get_labels()):.4f}')

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, davies_bouldin_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Davies-Bouldin Score')
    plt.title('Davies-Bouldin Analysis for PAM')
    plt.show()
def PAM_Inertia_analysis(data, k_range,print_scores=False):
    inertia_scores = []
    for k in k_range:
        model = PAM(data,n_clusters=k)
        model.fit()
        inertia_scores.append(model.get_Inertia())
        if print_scores:
            print(f'k={k}, Inertia: {model.get_Inertia():.4f}')

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Inertia Analysis for PAM')
    plt.show()