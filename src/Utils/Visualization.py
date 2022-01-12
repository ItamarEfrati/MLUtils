import itertools

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

from kneed import KneeLocator
from matplotlib.axes import Axes
from sklearn.manifold import TSNE


def plot_pie_relapse_per_cluster(counts, number_of_plots_per_row=4):
    number_of_clusters = len(counts.index.levels[0])
    n_rows = math.ceil(number_of_clusters / number_of_plots_per_row)
    n_cols = number_of_plots_per_row
    f, axs = plt.subplots(n_rows, n_cols, figsize=(n_rows * 8, n_cols * 2))

    for c_num in range(n_rows * n_cols):
        ax = axs[int(c_num / 4), c_num % 4]
        if c_num not in range(number_of_clusters):
            ax.axis('off')
            continue
        ax.pie(counts.loc[c_num], labels=counts.loc[c_num].index, autopct='%1.1f%%')
        ax.legend(loc='upper right', labels=counts.loc[c_num].values)
        ax.set_title(f"Cluster {c_num + 1}")

    plt.show()


def plot_clusters_analysis(kmeans, data_df, features, label_df):
    """
    plots the analysis of a single kmeans
    :param kmeans:
    :param data_df:
    :param features:
    :param label_df:
    :return:
    """
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # plot 1
    clusters_count_per_subject = data_df.groupby(['Subject', 'clusters']).count().iloc[:, 0] \
        .reset_index('clusters') \
        .groupby('Subject') \
        .clusters.nunique() \
        .value_counts()

    my_labels = clusters_count_per_subject.index

    ax = fig.add_subplot(gs[0, 0])
    ax.pie(clusters_count_per_subject, labels=my_labels)
    ax.set_title("Number of clusters per subject")

    # plot 2
    ax = fig.add_subplot(gs[0, 1])
    ax.pie(label_df['clusters'].value_counts(), labels=label_df['clusters'].value_counts().index)
    ax.set_title("Subjects majority cluster")

    # plot 3
    dists = kmeans.transform(features)
    dist2centroid = dists.min(axis=1)

    ax = fig.add_subplot(gs[1, :])
    dist2centroid.sort()
    ax.plot(dist2centroid)
    ax.set_title("Distance to centroid")
    plt.show()


def plot_cluster_2d(features, labels, n_clusters, ax=None):
    """
    plots a given cluster labels in 2d using T-SNE for dimension reduction
    :param ax:
    :param features:
    :param labels:
    :param n_clusters:
    """
    if not ax:
        _, ax = plt.subplots(figsize=(15, 15))
    two_dim_features = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(features)
    sns.scatterplot(two_dim_features[:, 0], two_dim_features[:, 1], hue=labels, legend='full', ax=ax,
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

    plot_simple_graph(clusters_range, sse, x_label="Number of Clusters", y_label='SSE', ax=axs[0], title="SSE results")
    plot_simple_graph(clusters_range, silhouette_coefficients, x_label="Number of Clusters",
                      y_label="Silhouette Coefficient", ax=axs[1], title="Silhouette results")

    plt.tight_layout()

    plt.show()
    kl = KneeLocator(clusters_range, sse, curve="convex", direction="decreasing")

    print("The elbow is at", kl.elbow)


def plot_bic_scores(bic_scores, clusters_range, covariance_types, ax=None):
    """

    :param bic_scores:
    :param clusters_range:
    :param covariance_types:
    :param ax:
    :return:
    """
    bars = []
    bic_scores = np.array(bic_scores)
    color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
    plt.figure(figsize=(8, 6))
    spl = plt.subplot(2, 1, 1)
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
    plt.ylim([bic_scores.min() * 1.01 - 0.01 * bic_scores.max(), bic_scores.max()])
    plt.title("BIC score per model")
    xpos = (
            np.mod(bic_scores.argmin(), len(clusters_range))
            + 0.65
            + 0.2 * np.floor(bic_scores.argmin() / len(clusters_range))
    )
    plt.text(xpos, bic_scores.min() * 0.97 + 0.03 * bic_scores.max(), "*", fontsize=14)
    spl.set_xlabel("Number of components")
    spl.legend([b[0] for b in bars], covariance_types)
    plt.show()


def plot_simple_graph(x, y, x_label, y_label, title, ax: Axes = None):
    """
    given 2d data plots a simple graph where the indices of the data are the x axis and the values of the
    data are the y axis.
    :param title:
    :param ax:
    :param data:
    :param x_label: the label of the x axis
    :param y_label: the label of the y axis
    :return:
    """
    has_ax = ax is not None
    if not has_ax:
        _, ax = plt.figure()
    ax.plot(x, y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.title.set_text(title)
    if not has_ax:
        plt.show()


def plot_graphs(data_dict, n_rows, n_columns, x_label, y_label):
    """

    :param data_dict: key is title and value is the data to plot
    :return:
    """
    f, axs = plt.subplots(n_rows, n_columns, figsize=(8 * n_rows, 8 * n_columns))

    for i, ax in enumerate(axs.reshape(-1)):
        current_values = list(data_dict.items())[i]
        plot_simple_graph(current_values[1][0], current_values[1][1], x_label=x_label, y_label=y_label,
                          title=current_values[0], ax=ax)
    plt.show()
