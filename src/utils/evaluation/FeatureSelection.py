import itertools
from abc import ABC, abstractmethod
from itertools import accumulate

import pandas as pd
import numpy as np

from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from mlxtend.feature_selection import SequentialFeatureSelector as SFS


class ColumnsNames:
    features_indices_column = "feature_num"
    features_name_column = "feature"
    scoring_column = "score"
    sum_rank_column = "sum_rank"
    sum_indicator_column = "sum_index"


# region Variance Feature Selection

def get_variance_features(data_df: pd.DataFrame, variance_threshold=(.99 * (1 - .99))):
    """
    scale data in [0,1] (sklearn MinMaxScaler) and apply sklearn VarianceThreshold to remove features (columns)
    with variance lower the threshold
    :param data_df: data-frame of the data to remove low variance features (columns)
    :param variance_threshold: threshold for variance with default value (.99 * (1 - .99))
    :return: a tuple of the selected feature indices and their names
    """
    variance_selector = VarianceThreshold(threshold=variance_threshold)
    min_max_scaler = MinMaxScaler()
    scaled_data = min_max_scaler.fit_transform(data_df)
    variance_selector.fit_transform(scaled_data)
    selected_features_indices = variance_selector.get_support(indices=True)
    print(f"VarianceThreshold selected {selected_features_indices.shape[0]} out of {data_df.shape[1]} features")
    return selected_features_indices, data_df.iloc[:, selected_features_indices].columns


# endregion

# region Univariate feature selection

class UnivariateScore:

    def __init__(self, method, features_indices, features_names):
        self.method = method
        self.features_indices = features_indices
        self.features_names = features_names

    def compute_features_score(self, data_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
        """
        Seek uni-variate relation strength between features and target by applying sk-learn SelectKBest to calculate
        importance of all features (columns) in the input data
        :param data_df: data-frame of the data
        :param label_df: data-frame of the label
        :return: pandas series with the data features importance
        """
        select_k_best = SelectKBest(score_func=self.method, k='all')
        select_k_best.fit(data_df, np.ravel(label_df.to_numpy()))
        features_score = select_k_best.scores_
        features_scores_df = pd.DataFrame(zip(self.features_indices, self.features_names, features_score))
        features_scores_df.columns = [ColumnsNames.features_indices_column, ColumnsNames.features_name_column,
                                      ColumnsNames.scoring_column]
        return features_scores_df


def get_univariate_feature_indices(data_df, label_df, features_indices, univariate_methods_list, num_features_to_select,
                                   is_num_is_top, top_univariate):
    """
    Select features using univariate methods by computing the feature importance for each method and combine the ranks.
    :param data_df: data-frame of the data
    :param label_df: data-frame of the label
    :param features_indices: the original indices of the feature
    :param univariate_methods_list: a list of sklearn feature selection methods
    :param num_features_to_select: number of features to select
    :param is_num_is_top: True if num of features to select is maximum otherwise the num is minimum
    :param top_univariate:
    :return: the original indices of the selected features
    """
    univariate_scores_list = list(
        map(lambda x: UnivariateScore(x, features_indices, data_df.columns), univariate_methods_list))
    univariate_scores_list = list(map(lambda x: x.compute_features_score(data_df, label_df), univariate_scores_list))

    features_rank_df = get_features_rank_by_score(univariate_scores_list, num_features_to_select)
    selected_univariate_features_df = selected_features_by_rank(features_rank_df, top_univariate, is_num_is_top)
    selected_univariate_indices = list(selected_univariate_features_df[ColumnsNames.features_indices_column])

    print(f"Custom univariate selected {len(selected_univariate_indices)} out of {data_df.shape[1]} features")
    return selected_univariate_indices


# endregion

# region Multivariate feature selection

def get_multivariate_class(method_name, params):
    if method_name in "decision_tree":
        return DecisionTreeMultivariateScore(params[0], params[1], params[2])
    if method_name in "sfs":
        return SequentialForwardSelectionMultivariateScore(params[0], params[1], params[2])
    return None


class MultivariateScore(ABC):

    # Todo check min_to_select
    def __init__(self, num_features_to_select, features_indices, features_names):
        self.min_to_select = num_features_to_select
        self.features_indices = features_indices
        self.features_names = features_names

    @abstractmethod
    def run_method(self, data_df, label_df, cv, groups):
        """
        Run the multivariate method over the data and creates a rank dataframe for the features
        :param groups:
        :param data_df: features dataframe
        :param label_df: labels dataframe
        :param cv: cross validation object
        :return: rank dataframe
        """
        pass


class DecisionTreeMultivariateScore(MultivariateScore):

    @staticmethod
    def choose_best_ccp(summary_df):
        """
        use the lowest ccp_alpha in the highest cv_accuracy category
        if this ccp_alpha is the max at the category it means that the entire tree is pruned
        in this case take the 2nd lowest ccp_alpha that correspond to the 2nd highest cv_accuracy
        :param summary_df:
        :return: best ccp alpha
        """
        grouped = summary_df.groupby('cv_accuracy')
        if (grouped.max() - grouped.min()).iloc[-1, 0] != 0:
            best_ccp_alpha = grouped.min().iloc[-1, 0]
        else:
            best_ccp_alpha = grouped.min().iloc[-2, 0]

        return best_ccp_alpha

    def find_best_tree(self, data_df: pd.DataFrame, label_df: pd.DataFrame, ccp_alphas: list):
        accuracies_list = []
        trees_dict = {}
        for ccp_alpha in ccp_alphas:
            current_decision_tree = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
            current_decision_tree.fit(data_df, label_df)
            accuracies_list.append(current_decision_tree.score(data_df, label_df))
            trees_dict[ccp_alpha] = current_decision_tree

        summary_df = pd.DataFrame([ccp_alphas, accuracies_list]).transpose()
        summary_df.columns = ['ccp_alpha', 'cv_accuracy']
        summary_df.sort_values(by=['cv_accuracy', 'ccp_alpha'], ascending=[False, False], inplace=True)
        best_ccp_alpha = self.choose_best_ccp(summary_df)
        return trees_dict[best_ccp_alpha]

    def run_method(self, data_df: pd.DataFrame, label_df: pd.DataFrame, cross_validation, groups):
        print("Starting Decision Tree multivariate feature selection process")
        ls_fold = []
        for fold_num, (train_index, test_index) in enumerate(cross_validation.split(data_df, label_df, groups)):
            print(f"Evaluating fold {fold_num}")
            decision_tree = DecisionTreeClassifier()
            current_data_df, current_label_df = data_df.iloc[train_index], label_df.iloc[train_index]
            ccp_alphas = decision_tree.cost_complexity_pruning_path(current_data_df, current_label_df)["ccp_alphas"]
            best_tree = self.find_best_tree(current_data_df, current_label_df, ccp_alphas)
            best_tree_score_df = pd.DataFrame.from_dict(
                {ColumnsNames.features_indices_column: self.features_indices,
                 ColumnsNames.features_name_column: self.features_names,
                 ColumnsNames.scoring_column: best_tree.feature_importances_})

            ls_fold.append(best_tree_score_df)

        select_rank_tree_df = get_features_rank_by_score(ls_fold, self.min_to_select, weighting=True)

        return select_rank_tree_df


class SequentialForwardSelectionMultivariateScore(MultivariateScore):

    def __init__(self, num_features_to_select, features_indices, features_names):
        super().__init__(num_features_to_select, features_indices, features_names)
        self.sfs_clf_ls = [SVC(kernel='linear'), GaussianNB(), LogisticRegression()]

    def summarize_results(self, classifiers_results: list):
        """
        combine the selection indicators and ranks of the features according to the given classifiers
        :param classifiers_results: list of lists - a list per classifier. a classifier list includes the following SFS items:
            0: k_feature_idx_ - Feature Indices of the selected feature subsets,
            1: k_feature_names_ - Feature names of the selected feature subsets,
            2: k_score_ - Cross validation average score of the selected subset,
            3: subsets_ - dictionary with MLxtend sequential forward selection subsets,
            4: list of ordered indices of the features according to their introduction order to the model
        :return: list of two data-frames.
            the first data-frame with feature indices, names and selector indicator column per classifier that represents
            whether the feature was selected according to that classifer.
            the second data-frame with feature indices, names and rank column per classifier that represents
            the order when the feature was introduced to the model according to that classifer.
        """
        zeros = np.zeros(len(self.features_indices))
        columns = [ColumnsNames.features_indices_column, ColumnsNames.features_name_column,
                   ColumnsNames.sum_indicator_column, ColumnsNames.sum_rank_column]
        summary_df = pd.DataFrame(zip(self.features_indices, self.features_names, zeros, zeros), columns=columns)

        for i, classifier_results in enumerate(classifiers_results):
            summary_df.loc[classifier_results[0], ColumnsNames.sum_indicator_column] += 1
            features_ranks = dict(zip(classifier_results[1], range(1, 1 + len(classifier_results[1]))))
            summary_df[ColumnsNames.sum_rank_column] = summary_df.apply(
                lambda x: x[ColumnsNames.sum_rank_column] + features_ranks[x[ColumnsNames.features_name_column]],
                axis=1)

        return summary_df

    @staticmethod
    def ordered_features(subsets: dict) -> list:
        """
        get MLxtend sequential forward selection subsets dict and return a list with the features indices ordered
        according to their addition to the model
        :param subsets: dictionary with MLxtend sequential forward selection subsets
            A dictionary of selected feature subsets during the sequential selection,
            where the dictionary keys are the lengths k of these feature subsets. The dictionary values are
            dictionaries themselves with the following keys: 'feature_idx' (tuple of indices of the feature subset)
            'feature_names' (tuple of feature names of the feat. subset)
            'cv_scores' (list individual cross-validation scores)
            'avg_score' (average cross-validation score)
        :return: list of feature indices according to the order they were added to the model
        """
        subsets_df = pd.DataFrame.from_dict(subsets)
        ls = [[*set(subsets_df.iloc[3, 0]), ]]
        for col_index in range(len(subsets_df.columns) - 1):
            set1 = set(subsets_df.iloc[3, col_index])
            set2 = set(subsets_df.iloc[3, col_index + 1])
            diff = set2.difference(set1)
            ls.append([*diff, ])
        merged = list(itertools.chain.from_iterable(ls))
        return merged

    def get_sequential_forward_selection_reults(self, data_df, label_df, k_feature_range: str, cv,
                                                groups) -> pd.DataFrame:
        """
        create a data-frame with feature indices, names and relative importance that represents the multi-variate
        relation strength to the target according to sequential feature selection (SFS) methods.Execute several SFS
        corresponding to the given classifiers list and combining the corresponding feature importance, selection
        indicator, and rank
        :param cv: cross validation object
        :param data_df: data-frame of data to calculate it's features (columns) importance. columns names are indices
        :param label_df: data-frame of data to explore multi-variate relation with each feature
        :param k_feature_range: interval range to limit the forward selection. could also be 'best' or 'parsimonious'
        :return: list of lists - a list per classifier. a classifier list includes the following SFS items:
        0: k_feature_idx_ - Feature Indices of the selected feature subsets,
        1: k_feature_names_ - Feature names of the selected feature subsets,
        2: k_score_ - Cross validation average score of the selected subset,
        3: subsets_ - dictionary with MLxtend sequential forward selection subsets,
        4: list of ordered indices of the features according to their introduction order to the model
        """
        print("Running sequential forward selection")
        list_ls = []
        for i, clf in enumerate(self.sfs_clf_ls):
            print(f"Running classifier number {i + 1}")
            sfs = SFS(clf, k_features=k_feature_range, forward=True, floating=False, verbose=1, cv=cv, n_jobs=-1)
            sfs.fit(data_df, label_df, groups=groups)
            running_results = [list(sfs.k_feature_idx_), self.ordered_features(sfs.subsets_)]
            list_ls.append(running_results)
        return self.summarize_results(list_ls)

    def run_method(self, data_df, label_df, cv, groups):
        scaled_data_df = pd.DataFrame(StandardScaler().fit_transform(data_df), columns=data_df.columns)
        sfs_results_df = self.get_sequential_forward_selection_reults(scaled_data_df, label_df.to_numpy().reshape(-1),
                                                                      'parsimonious', cv, groups)
        sfs_results_df.sort_values(by=[ColumnsNames.sum_indicator_column, ColumnsNames.sum_rank_column],
                                   ascending=[False, True], inplace=True)
        return sfs_results_df


def combine_multivariate_feature_selection(dfs_list):
    columns_to_sum = [ColumnsNames.sum_indicator_column, ColumnsNames.sum_rank_column]
    for df in dfs_list:
        df.sort_values(by=[ColumnsNames.features_indices_column], ascending=[True], inplace=True)

    summary_df = dfs_list[0].copy()
    for df in dfs_list[1:]:
        summary_df.loc[:, columns_to_sum] = summary_df[columns_to_sum].values + df[columns_to_sum].values

    return summary_df


def get_multivariate_feature_indices(data_df, label_df, features_indices, num_features_to_select,
                                     multivariate_methods_dict, cross_validation, groups):
    """

    :param data_df:
    :param label_df:
    :param features_indices:
    :param num_features_to_select:
    :param multivariate_methods_dict:
    :param cross_validation:
    :param groups:
    :return:
    """
    multivariate_features_ranks = []
    for method_name, params in multivariate_methods_dict.items():
        params = params + [features_indices, data_df.columns]
        multivariate_method_class = get_multivariate_class(method_name, params)
        multivariate_features_ranks.append(
            multivariate_method_class.run_method(data_df, label_df, cross_validation, groups))
    ranked_features_df = combine_multivariate_feature_selection(multivariate_features_ranks)
    selected_features_df = selected_features_by_rank(ranked_features_df, num_features_to_select, is_num_is_minimum=True)

    return selected_features_df[ColumnsNames.features_indices_column].to_numpy()


# endregion

# region custom selection

def custom_feature_selection(data_df, label_df, cross_validation, groups, univariate_methods_list, min_to_select,
                             top_univariate, top_multivariate, multivariate_methods_dict):
    chosen_variance_features_indices, chosen_variance_features_names = get_variance_features(data_df)
    variance_data_df = data_df.iloc[:, chosen_variance_features_indices]

    selected_univariate_indices = get_univariate_feature_indices(variance_data_df, label_df,
                                                                 chosen_variance_features_indices,
                                                                 univariate_methods_list, min_to_select,
                                                                 is_num_is_top=True,
                                                                 top_univariate=top_univariate)

    univariate_features_df = data_df.iloc[:, selected_univariate_indices]

    multivariate_methods_dict = {k: [min_to_select] + v for k, v in multivariate_methods_dict.items()}
    multivariate_features_indices = get_multivariate_feature_indices(univariate_features_df, label_df,
                                                                     selected_univariate_indices,
                                                                     top_multivariate, multivariate_methods_dict,
                                                                     cross_validation, groups)

    return data_df.iloc[:, multivariate_features_indices].columns


# endregion

# region feature importance combinations

def get_features_rank_by_score(feature_selection_scores_list: list, min_to_select: int = 10,
                               weighting: bool = False) -> pd.DataFrame:
    """
    create a data-frame with original indices, features name and relative importance that represents the relation
    strength to the target and combining the corresponding feature importance, selection indicator and rank.
    :param weighting: true means that the summation of the ranks will be weighted
    :param feature_selection_scores_list: list of features importance from methods
    :param min_to_select: minimum features to select in each method
    :return: df_select_rank: data-frame which is sorted first in descending order of the sum number of methods
        that selected the feature and second in ascending order of their sum of ranks
    """
    features_combine_score_df = feature_selection_scores_list[0].drop([ColumnsNames.scoring_column], axis=1)
    features_combine_score_df[ColumnsNames.sum_indicator_column] = 0
    features_combine_score_df[ColumnsNames.sum_rank_column] = 0

    weight = 1 / len(feature_selection_scores_list) if weighting else 1

    for feature_selection_scores in feature_selection_scores_list:
        order_scores_df = feature_selection_scores.sort_values(by=[ColumnsNames.scoring_column], ascending=False,
                                                               ignore_index=True)
        order_scores_df['rank'] = 1 + np.arange(order_scores_df.shape[0])

        scores_sum = order_scores_df[ColumnsNames.scoring_column].sum()
        # select features with 95% of the accumulated score
        accumulate_threshold = 0.95 * scores_sum
        accumulate_score_df = pd.DataFrame(accumulate(order_scores_df[ColumnsNames.scoring_column]))
        first_condition_indices = set(np.where(accumulate_score_df < accumulate_threshold)[0])

        # selected features that have score with at least 1% of total accumulated score
        minimum_feature_score = 0.01 * scores_sum
        second_condition_indices = set(
            np.where(order_scores_df[ColumnsNames.scoring_column] > minimum_feature_score)[0]
        )

        selected_indices = first_condition_indices.intersection(second_condition_indices)

        # Todo check else condition
        # sanity check - minimum of features selected
        selected_indices = selected_indices if len(selected_indices) >= min_to_select \
            else order_scores_df['rank'] <= min_to_select

        order_scores_df['ind'] = False
        order_scores_df.loc[selected_indices, 'ind'] = True

        order_scores_df = order_scores_df.sort_values(by=[ColumnsNames.features_indices_column],
                                                      ascending=True, ignore_index=True)

        features_combine_score_df[ColumnsNames.sum_indicator_column] += weight * order_scores_df['ind']
        features_combine_score_df[ColumnsNames.sum_rank_column] += weight * order_scores_df['rank']

    features_combine_score_df.sort_values(by=[ColumnsNames.sum_indicator_column, ColumnsNames.sum_rank_column],
                                          ascending=[False, True], inplace=True, ignore_index=True)

    return features_combine_score_df


def selected_features_by_rank(ranks_df: pd.DataFrame, num_features_to_select: int = 50, is_num_is_minimum: bool = True):
    """
    Select the minimum between input number and the most influential features (at least in one method this feature was
    selected)
    :param ranks_df: data-frame which is sorted first in descending order of the sum number of methods
        that selected the feature and second in ascending order of their sum of ranks
    :param num_features_to_select: int that determine the number of first row of df_select_rank to return
    :param is_num_is_minimum: str that determines whether to select minimum or maximum top_features
    :return: data-frame containing only the selected features.
    """
    indices = ranks_df[ColumnsNames.sum_indicator_column] >= 1
    if is_num_is_minimum:
        number = np.minimum(num_features_to_select, np.sum(indices))
    else:
        number = np.maximum(num_features_to_select, np.sum(indices))
    return ranks_df[[ColumnsNames.features_indices_column, ColumnsNames.features_name_column]].head(number)

# endregion
