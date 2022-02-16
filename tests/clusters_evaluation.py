import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from Constants import PREPROCESS_FOLDER, DATA_FOLDER
from evaluation.Clustering import evaluate_kmeans

if __name__ == '__main__':
    preprocess_folder = os.path.join(DATA_FOLDER, PREPROCESS_FOLDER)
    merge_df = pd.read_csv(os.path.join(preprocess_folder, "merge_data.csv"), index_col=['Id', 'Date'],
                           parse_dates=['Date'])
    label_df = merge_df['is_placebo']
    merge_df = merge_df.drop(columns=['is_placebo'])
    merge_features = merge_df.values
    scaled_merge_features = MinMaxScaler().fit_transform(merge_features)

    kmeans_kwargs = {
        "init": "k-means++",
        "n_init": 50,
        "max_iter": 1000,
        "random_state": 42
    }

    evaluate_kmeans(scaled_merge_features, clusters_range=range(2, 20), njobs=-1, kmeans_kwargs=kmeans_kwargs)
    #evaluate_gmm(scaled_merge_features, clusters_range=range(2, 10), njobs=3)

    # t = [[1, 2, 3], [4, 5, 6]]
    # plt.plot(*zip(*t))
    # plt.show()
