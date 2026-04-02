from Algorithm import CLARANS
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
def clarans_fast_build(data,n_clusters,num_local,max_neighbors,first_column,second_column,data_pca):

    model=CLARANS(data,n_clusters,num_local,max_neighbors)
    model.fit()
    labels = [0] * len(data)

    clusters=model.get_clusters()
    medoids=model.get_medoids()
    for cluster_id, cluster in enumerate(clusters):
        for index in cluster:
            labels[index] = cluster_id
    data['cluster'] = labels
    
    model.sihouette_score_= silhouette_score(data.drop(columns=['cluster']), labels)
    print(f'Silhouette Score: {model.sihouette_score_:.4f}')

    medoid_points = data.iloc[medoids]
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(data[first_column],
            data[second_column],
            c=data['cluster'])

    plt.scatter(medoid_points[first_column],
                medoid_points[second_column],
                marker='s',
                s=100,color='red'
                )

    plt.xlabel(f"{first_column}")
    plt.ylabel(f"{second_column}")
    plt.title("CLARANS Customer Segmentation")

    plt.subplot(1, 2, 2)
    plt.scatter(data_pca[:,0], data_pca[:,1], c=labels)
    plt.scatter(medoid_points[first_column],
                medoid_points[second_column],
                marker='s',
                s=100,color='red'
                )
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('CLARANS Clustering with PCA')
    plt.show()

def clarans_silhouette_analysis(data, k_range, num_local, max_neighbors):
    silhouette_scores = []
    for k in k_range:
        model = CLARANS(data, k, num_local, max_neighbors)
        model.fit()
        model.sihouette_score_=silhouette_score(data,model.get_labels())
        silhouette_scores.append(model.sihouette_score_)
        print(f'k={k}, Silhouette Score: {model.sihouette_score_:.4f}')

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for CLARANS')
    plt.show()