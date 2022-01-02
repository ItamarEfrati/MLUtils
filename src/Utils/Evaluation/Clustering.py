from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from Utils.Visualization import Visualization


def calculate_kmeans(features, kmeans_kwargs, clusters_range):
    print("Features shape", features.shape)
    # A list holds the SSE values for each k
    sse = []
    silhouette_coefficients = []
    kmeans_dict = {}
    for k in tqdm(clusters_range):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(features)
        kmeans_dict[k] = kmeans
        sse.append(kmeans.inertia_)
        score = silhouette_score(features, kmeans.labels_)
        silhouette_coefficients.append(score)

    Visualization.plot_silhouette_and_sse(clusters_range, sse, silhouette_coefficients)

    return kmeans_dict
