import seaborn as sns
import matplotlib.pyplot as plt
import math

from kneed import KneeLocator
from sklearn.manifold import TSNE


class Visualization:

    @staticmethod
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

    @staticmethod
    def plot_kmeans_analysis(kmeas, data_df, features, label_df):
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
        dists = kmeas.transform(features)
        dist2centroid = dists.min(axis=1)

        ax = fig.add_subplot(gs[1, :])
        dist2centroid.sort()
        ax.plot(dist2centroid)
        ax.set_title("Distance to centroid")
        plt.show()

    @staticmethod
    def plot_cluster_2d(features, labels, n_clusters, random_state):
        two_dim_features = TSNE(n_components=2, learning_rate='auto', init='random',
                                random_state=random_state).fit_transform(features)
        sns.scatterplot(two_dim_features[:, 0], two_dim_features[:, 1], hue=labels, legend='full',
                        palette=sns.color_palette("bright", n_clusters))
        plt.show()

    @staticmethod
    def plot_silhouette_and_sse(clusters_range, sse, silhouette_coefficients):
        f, axs = plt.subplots(2, 1, figsize=(16, 8))

        # plot 1
        ax = axs[0]
        ax.plot(clusters_range, sse)
        ax.set_xticks(clusters_range)
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("SSE")

        # plot 2
        ax = axs[1]
        ax.plot(clusters_range, silhouette_coefficients)
        ax.set_xticks(clusters_range)
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Silhouette Coefficient")

        plt.show()
        kl = KneeLocator(clusters_range, sse, curve="convex", direction="decreasing")

        print("The elbo is at", kl.elbow)
