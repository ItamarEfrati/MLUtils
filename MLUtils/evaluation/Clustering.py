import itertools
import numpy as np
import pandas as pd

import MLUtils.Visualization as Visualization
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.manifold import TSNE

from kneed import KneeLocator
from functools import reduce

from MLUtils.MultiProcess import run_function_parallel

# region Cluster default arguments

KMEANS_KWARG = {
    "init": "k-means++",
    "n_init": 50,
    "max_iter": 1000,
    "random_state": 42
}


# endregion

# region Evaluations

def _evaluate_single_kmeans(*args):
    """
    Evaluate single kmeans
    :param args: a list of arguments. First item is dict arguments for kmeans
    and second argument are the data for training
    :return: the trained kmeans object
    """
    kmeans = KMeans(**args[0][0])
    kmeans.fit_predict(args[0][1])
    return kmeans


def evaluate_kmeans(features, kmeans_kwargs, clusters_range: range, njobs):
    """
    Runs a full evaluation of kmeans with different arguments and plots the silhouette and the SSE scores graphs
    in order to choose the best parameters
    :param features: the data to run k-means over.
    :param kmeans_kwargs: the arguments to pass to each kmeans object without n_clusters
    :param clusters_range: the range of clusters to test
    :param njobs: number of jobs to run the evaluations, -1 means max.
    :return: a dictionary where the key are the number of components and the value in the relevant kmeans object
    """
    args_combinations = [(kmeans_kwargs.copy(), i, features) for i in clusters_range]
    list(map(lambda k: k[0].update({"n_clusters": k[1]}), args_combinations))
    args_combinations = list(map(lambda d: (d[0], d[-1]), args_combinations))

    kmeans_list = run_function_parallel(_evaluate_single_kmeans, njobs, *args_combinations)
    kmeans_dict = {g.n_clusters: g for g in kmeans_list}
    silhouette_coefficients = list(
        map(lambda g: silhouette_score(features, g.labels_, metric='euclidean'), kmeans_dict.values()))
    sse = list(map(lambda g: g.inertia_, kmeans_dict.values()))
    plot_kmeans_evaluation_results(clusters_range, sse, silhouette_coefficients)

    return kmeans_dict


# Todo change to hdbscan
def evaluate_dbscan(features, min_samples):
    """
    Runs a dbscan evaluation process
    :param features:
    :param min_samples:
    :return:
    """
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

    plot_cluster_2d(features, cluster_labels, len(set(cluster_labels)))

    return dbscan


def _evaluate_single_gmm(*args):
    """
    Evaluate single GMM
    :param args: a list of arguments. First item is dict arguments for gmm
    and second argument are the data for training
    :return: the trained kmeans object and it's label for the given data
    """
    gmm = GaussianMixture(**args[0][0])
    labels = gmm.fit_predict(X=args[0][1])
    return gmm, labels


def evaluate_gmm(features, gmm_kwargs, components_range, njobs=-1):
    """
    Runs full evaluation of GMM given gmm arguments with combination of components range and covariance types.
    Plots the silhouette score and the bic score of the evaluation for each covariance type
    :param features: the data for training
    :param gmm_kwargs: dict with gmm arguments to pass for all gmm objects
    :param components_range: the range of components to test
    :param njobs: the number of jobs to run in parallel in the evaluation part, -1 is maximum.
    :return: a dictionary where the keys are the covariance type and number of components
    and the value is the relevant gmm object
    """
    covariance_types = sorted(["spherical", "tied", "diag", "full"])
    args_combinations = [(gmm_kwargs.copy(), i, j, features) for i in components_range for j in covariance_types]
    list(map(lambda k: k[0].update({"n_components": k[1], "covariance_type": k[2]}), args_combinations))
    args_combinations = list(map(lambda d: (d[0], d[-1]), args_combinations))

    gmms_list = run_function_parallel(_evaluate_single_gmm, njobs, *args_combinations)
    gmms_dict = {(g[0].n_components, g[0].covariance_type): g for g in gmms_list}
    gmms_dict = dict(sorted(gmms_dict.items()))
    silhouette_coefficients = list(
        map(lambda g: silhouette_score(features, g[1], metric='euclidean'), gmms_dict.values()))

    bic_scores_list = list(map(lambda g: g[0].bic(features), gmms_dict.values()))
    minimum_arg_index = plot_bic_scores(bic_scores_list, components_range, covariance_types)
    print("Minimum bic score is for", args_combinations[minimum_arg_index][0])

    silhouette_coefficients_dict = {}
    for i, covariance_type in enumerate(covariance_types):
        silhouette_coefficients_dict[covariance_type] = [components_range,
                                                         silhouette_coefficients[i::len(covariance_types)]]

    x_label = ['number of components'] * len(silhouette_coefficients_dict)
    y_label = ['silhouette score'] * len(silhouette_coefficients_dict)
    Visualization.plot_graphs(silhouette_coefficients_dict, n_rows=2, n_columns=2, x_label=x_label, y_label=y_label)

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
        feature_importance_df.columns = [f"cluster {i} features", f"cluster {i} features scores"]
        features_importance_list.append(feature_importance_df)

    return reduce(lambda df1, df2: pd.merge(df1, df2, left_index=True, right_index=True), features_importance_list)


# endregion

# region Visualization

def plot_cluster_2d(features, labels, n_clusters, ax=None, reduction_algorithm='TSNE'):
    """
    plots a given cluster labels in 2d using T-SNE for dimension reduction
    :param reduction_algorithm:
    :param ax:
    :param features:
    :param labels:
    :param n_clusters:
    """
    if not ax:
        _, ax = plt.subplots(figsize=(15, 15))
    reductions_dict = {"TSNE": TSNE(n_components=2, learning_rate='auto', init='random')}
    two_dim_features = reductions_dict[reduction_algorithm].fit_transform(features)
    sns.scatterplot(x=two_dim_features[:, 0], y=two_dim_features[:, 1], hue=labels, legend='full', ax=ax,
                    palette=sns.color_palette("bright", n_clusters))

    plt.show()


def plot_kmeans_evaluation_results(clusters_range, sse, silhouette_coefficients):
    """
    plots the sum of the squared distance and the silhouette scores of the evaluation of kmeans
    :param clusters_range:
    :param sse:
    :param silhouette_coefficients:
    :return:
    """
    f, axs = plt.subplots(2, 1, figsize=(17, 8))

    Visualization.plot_simple_graph(clusters_range, sse, x_label="Number of Clusters", y_label='SSE', ax=axs[0],
                                    title="SSE results")
    Visualization.plot_simple_graph(clusters_range, silhouette_coefficients, x_label="Number of Clusters",
                                    y_label="Silhouette Coefficient", ax=axs[1], title="Silhouette results")

    plt.tight_layout()

    plt.show()
    kl = KneeLocator(clusters_range, sse, curve="convex", direction="decreasing")

    print("The elbow is at", kl.elbow)


def plot_bic_scores(bic_scores, clusters_range, covariance_types):
    """

    :param bic_scores:
    :param clusters_range:
    :param covariance_types:
    :return:
    """
    bars = []
    bic_scores = np.array(bic_scores)
    color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
    plt.figure(figsize=(8, 6))
    subplots = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(covariance_types, color_iter)):
        xpos = np.array(clusters_range) + 0.2 * (i - 2)
        bars.append(
            plt.bar(
                xpos,
                bic_scores[i * len(clusters_range): (i + 1) * len(clusters_range)],
                width=0.2,
                color=color,
            )
        )
    plt.xticks(clusters_range)
    subplots.set_xlabel("Number of components")
    subplots.legend([b[0] for b in bars], covariance_types)
    plt.show()
    return np.argmin(bic_scores)

# endregion
