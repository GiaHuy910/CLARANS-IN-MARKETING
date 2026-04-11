import seaborn as sns
import pandas as pd

from Algorithm import CLARANS
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.metrics import davies_bouldin_score
def clarans_fast_build(data,n_clusters,num_local,max_neighbors,first_column,second_column):

    model=CLARANS(data,n_clusters,num_local,max_neighbors)
    model.fit()

    medoids=model.get_medoids()
    labels=model.get_labels()
    data['cluster'] = labels
    
    model.sihouette_score_= silhouette_score(data.drop(columns=['cluster']), labels)
    print(f'Silhouette Score: {model.sihouette_score_:.4f}')

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[first_column],
            y=data[second_column],
            hue=data['cluster'], palette="Set2")

    for i, medoid in enumerate(medoids):
        plt.text(medoid[0], medoid[1], f'Medoid {i}', fontsize=9)
    plt.scatter(medoids[:, 0],
        medoids[:, 1],
        marker='s',
        s=100,color='red')
    plt.xlabel(first_column)
    plt.ylabel(second_column)
    plt.title("CLARANS Customer Segmentation")

    plt.show()

def clarans_silhouette_analysis(data, k_range, num_local, max_neighbors,print_scores=False):
    silhouette_scores = []
    for k in k_range:
        model = CLARANS(data, k, num_local, max_neighbors)
        model.fit()
        model.sihouette_score_=silhouette_score(data,model.get_labels())
        silhouette_scores.append(model.sihouette_score_)
        if print_scores:
            print(f'k={k}, Silhouette Score: {model.sihouette_score_:.4f}')

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for CLARANS')
    plt.show()

def clarans_davies_bouldin_analysis(data, k_range, num_local, max_neighbors,print_scores=False):
    db_scores = []
    for k in k_range:
        model = CLARANS(data, k, num_local, max_neighbors)
        model.fit()
        db_score = davies_bouldin_score(data, model.get_labels())
        db_scores.append(db_score)
        if print_scores:
            print(f'k={k}, Davies-Bouldin Score: {db_score:.4f}')

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, db_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Davies-Bouldin Score')
    plt.title('Davies-Bouldin Analysis for CLARANS')
    plt.show()
    
def clarans_Inertia_analysis(data, k_range, num_local, max_neighbors,print_scores=False):
    inertia_scores = []
    for k in k_range:
        model = CLARANS(data, k, num_local, max_neighbors)
        model.fit()
        inertia = model.get_Inertia()
        inertia_scores.append(inertia)
        if print_scores:
            print(f'k={k}, Inertia: {inertia:.4f}')

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Inertia Analysis for CLARANS')
    plt.show()