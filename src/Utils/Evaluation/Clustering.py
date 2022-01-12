from functools import reduce

import numpy as np
import pandas as pd
from kneed import KneeLocator

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier

import Utils.Visualization as Visualization
from Utils.MultiProcess import run_function_parallel


# Todo add comments and descriptions

# region Evaluations

def evaluate_single_kmeans_(*args):
    kmeans = KMeans(**args[0][0])
    kmeans.fit_predict(args[0][1])
    return kmeans


def evaluate_kmeans(features, kmeans_kwargs, clusters_range: range, njobs):
    print("Features shape", features.shape)
    args_combinations = [(kmeans_kwargs.copy(), i, features) for i in clusters_range]
    list(map(lambda k: k[0].update({"n_clusters": k[1]}), args_combinations))
    args_combinations = list(map(lambda d: (d[0], d[2]), args_combinations))

    kmeans_list = run_function_parallel(evaluate_single_kmeans_, njobs, *args_combinations)
    kmeans_dict = {g.n_clusters: g for g in kmeans_list}
    silhouette_coefficients = list(
        map(lambda g: silhouette_score(features, g.labels_, metric='euclidean'), kmeans_dict.values()))
    sse = list(map(lambda g: g.inertia_, kmeans_dict.values()))
    Visualization.plot_kmeans_evaluation_results(clusters_range, sse, silhouette_coefficients)

    return kmeans_dict


def evaluate_dbscan(features, min_samples):
    knn = NearestNeighbors(n_neighbors=min_samples).fit(features)
    distances, indices = knn.kneighbors(features)

    Visualization.plot_simple_graph(np.sort(distances[:, -1])[::-1], range(len(distances)), x_label='index',
                                    y_label='distance', title="di")

    kl = KneeLocator(range(indices.shape[0]), np.sort(distances[:, -1])[::-1], curve="convex", direction="decreasing")
    print(f"Elbow is {distances[:, -1][::-1][kl.elbow]}")
    best_eps = distances[:, -1][::-1][kl.elbow]

    print("Evaluating DBscan")
    dbscan = DBSCAN(eps=best_eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(features)
    print(f"There are in total {len(set(cluster_labels))} clusters")
    print(f"Silhouette score is {silhouette_score(features, cluster_labels)}")

    Visualization.plot_cluster_2d(features, cluster_labels, len(set(cluster_labels)))

    return dbscan


def evaluate_single_gmm_(*args):
    gmm = GaussianMixture(n_components=args[0][0], n_init=20, init_params='kmeans', covariance_type=args[0][1])
    labels = gmm.fit_predict(args[0][2])
    return gmm, labels


def evaluate_gmm(features, clusters_range, njobs=-1):
    covariance_types = sorted(["spherical", "tied", "diag", "full"])
    args_combinations = [(a, b, features) for b in covariance_types for a in clusters_range]

    gmms_list = run_function_parallel(evaluate_single_gmm_, njobs, *args_combinations)
    gmms_dict = {(g[0].n_components, g[0].covariance_type): g for g in gmms_list}
    gmms_dict = dict(sorted(gmms_dict.items()))
    silhouette_coefficients = list(
        map(lambda g: silhouette_score(features, g[1], metric='euclidean'), gmms_dict.values()))

    bic_scores_list = list(map(lambda g: g[0].bic(features), gmms_dict.values()))
    Visualization.plot_bic_scores(bic_scores_list, clusters_range, covariance_types)

    silhouette_coefficients_dict = {}
    for i, covariance_type in enumerate(covariance_types):
        silhouette_coefficients_dict[covariance_type] = [clusters_range,
                                                         silhouette_coefficients[i::len(covariance_types)]]

    Visualization.plot_graphs(silhouette_coefficients_dict, 2, 2, x_label='number of components',
                              y_label='silhouette score')

    return gmms_dict


# endregion

# region Cluster Analysis

def decision_tree_analysis(data: pd.DataFrame, labels: pd.Series):
    features_importance_list = []

    for i in range(len(set(labels))):
        dt = DecisionTreeClassifier()
        label = (labels == i).astype(int)
        dt.fit(data, label)
        feature_importance_df = pd.DataFrame([data.columns, dt.feature_importances_]).T
        feature_importance_df.sort_values(by=1, ascending=False, inplace=True)
        feature_importance_df.reset_index(inplace=True, drop=True)
        feature_importance_df.columns = [f"cluster {i} features", f"cluster {i}"]
        features_importance_list.append(feature_importance_df)

    return reduce(lambda df1, df2: pd.merge(df1, df2, left_index=True, right_index=True), features_importance_list)

# endregion
