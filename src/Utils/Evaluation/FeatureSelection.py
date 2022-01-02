import itertools
from abc import ABC, abstractmethod
from itertools import accumulate

import pandas as pd
import numpy as np

from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
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
    sum_index_column = "sum_index"


# region Variance Feature Selection

def get_variance_features(data_df: pd.DataFrame, variance_threshold=(.99 * (1 - .99))):
    """
    scale data in [0,1] (sk-learn MinMaxScaler) and apply sk-learn VarianceThreshold to remove features (columns)
    with variance lower the threshold
    :param data_df: data-frame of the data to remove low variance features (columns)
    :param variance_threshold: threshold for variance with default value (.99 * (1 - .99))
    :return: data-frame of data removed the low variance features and numpy array of the corresponding columns indices
    """
    variance_selector = VarianceThreshold(threshold=variance_threshold)
    min_max_scaler = MinMaxScaler()
    scaled_data = min_max_scaler.fit_transform(data_df)
    variance_selector.fit_transform(scaled_data)
    selected_features_indices = variance_selector.get_support(indices=True)
    print(f"VarianceThreshold selected {selected_features_indices.shape[0]} out of {data_df.shape[1]} features")
    return selected_features_indices, data_df.iloc[:, selected_features_indices].columns


# endregion

# region univariate feature selection

class UnivariateScore:

    def __init__(self, method, features_indices, features_names):
        self.method = method
        self.features_indices = features_indices
        self.features_names = features_names

    def compute_features_score(self, df_data: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
        """
        seek uni-variate relation strength between features and target by
        applying sk-learn SelectKBest to calculate importance of all features (columns) in the input data
        :param df_data: data-frame of the data
        :param target_df: data-frame of the target
        :return: pandas series with the data features importance
        """
        select_k_best = SelectKBest(score_func=self.method, k='all')
        select_k_best.fit(df_data.to_numpy(), np.ravel(target_df.to_numpy()))
        features_score = select_k_best.scores_
        features_scores_df = pd.DataFrame([self.features_indices, self.features_names, features_score]).transpose()
        features_scores_df.columns = [ColumnsNames.features_indices_column, ColumnsNames.features_name_column,
                                      ColumnsNames.scoring_column]
        return features_scores_df


def get_univariate_feature_indices(data_df, label_df, features_indices, features_names, univariate_methods_list,
                                   min_to_select, is_min_top_features, top_univariate):
    univariate_scores_list = list(
        map(lambda x: UnivariateScore(x, features_indices, features_names), univariate_methods_list))
    univariate_scores_list = list(map(lambda x: x.compute_features_score(data_df, label_df), univariate_scores_list))

    features_rank_df = get_features_rank(univariate_scores_list, min_to_select)
    selected_univariate_features_df = selected_rank_features(features_rank_df, top_univariate, is_min_top_features)
    selected_univariate_indices = list(selected_univariate_features_df[ColumnsNames.features_indices_column])

    print(f"Custom univariate selected {len(selected_univariate_indices)} out of {data_df.shape[1]} features")
    return selected_univariate_indices


# endregion

# region multivariate feature selection

def get_multivariate_class(method_name, params):
    if method_name in "decision_tree":
        return DecisionTreeMultivariateScore(params[0], params[1], params[2])
    if method_name in "sfs":
        return SequentialForwardSelectionMultivariateScore(params[0], params[1], params[2])
    return None


class MultivariateScore(ABC):

    def __init__(self, min_to_select, features_indices, features_names):
        self.min_to_select = min_to_select
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

        select_rank_tree_df = get_features_rank(ls_fold, self.min_to_select, weighting=True)
        columns = {
            ColumnsNames.sum_index_column: f'{ColumnsNames.sum_index_column}_tree',
            ColumnsNames.sum_rank_column: f'{ColumnsNames.sum_rank_column}_tree'
        }
        select_rank_tree_df = select_rank_tree_df.rename(columns=columns)

        return select_rank_tree_df


class SequentialForwardSelectionMultivariateScore(MultivariateScore):

    def __init__(self, min_to_select, features_indices, features_names):
        super().__init__(min_to_select, features_indices, features_names)
        self.sfs_clf_ls = [SVC(kernel='linear'), GaussianNB(), LogisticRegression(n_jobs=-1)]

    def create_select_indicator(self, train_data_df: pd.DataFrame, list_ls: list):
        """
        combine the selection indicators and ranks of the features according to the given classifiers
        :param train_data_df: data-frame of data to calculate it's features (columns) importance. columns names are indices
        :param list_ls: list of lists - a list per classifier. a classifier list includes the following SFS items:
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
        col_array = np.array(range(len(train_data_df.columns)))
        df: pd.DataFrame = pd.DataFrame(col_array, columns=[ColumnsNames.features_indices_column])
        df_rank: pd.DataFrame = df
        df_rank[ColumnsNames.features_name_column] = self.features_names
        for classifier_results in list_ls:
            indx = np.asarray(classifier_results[0])
            col = pd.Series(np.where(np.in1d(col_array, indx) == False, 0, 1))
            rank_array = np.zeros(col_array.shape, dtype=int)
            counter: int = 1
            for item in classifier_results[4]:
                row_indx: int = df[df[ColumnsNames.features_name_column] == item].index.values.astype(int).item()
                rank_array[row_indx] = counter
                counter = counter + 1
            df = pd.concat([df, col], axis=1)
            rank_col: pd.Series = pd.Series(rank_array)
            df_rank = pd.concat([df_rank, rank_col], axis=1)
        sum_col = pd.Series(df.iloc[:, 2:len(list_ls) + 2].sum(axis=1), name=ColumnsNames.sum_index_column)
        sum_rank = pd.Series(df_rank.iloc[:, 2:len(list_ls) + 2].sum(axis=1), name=ColumnsNames.sum_rank_column)
        df = pd.concat([df, sum_col], axis=1)
        df_rank = pd.concat([df_rank, sum_rank], axis=1)
        return df, df_rank

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

    def sequential_forward_selection(self, data_df, label_df, k_feature_range: str, cv, groups) -> pd.DataFrame:
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
            print(f"Running {i} classifier")
            sfs = SFS(clf, k_features=k_feature_range, forward=True, floating=False, verbose=1, cv=cv, n_jobs=-1)
            sfs.fit(data_df, label_df, groups=groups)
            running_results = [sfs.k_feature_idx_, sfs.k_feature_names_, sfs.k_score_, sfs.subsets_,
                               self.ordered_features(sfs.subsets_)]
            list_ls.append(running_results)
        df, df_rank = self.create_select_indicator(data_df, list_ls)
        return pd.merge(df, df_rank, how='left', on=ColumnsNames.features_indices_column)

    def run_method(self, data_df, label_df, cv, groups):
        scaled_data_df = pd.DataFrame(StandardScaler().fit_transform(data_df), columns=data_df.columns)
        df = self.sequential_forward_selection(scaled_data_df, label_df.to_numpy().reshape(-1),
                                               'parsimonious', cv, groups)
        df.sort_values(by=[ColumnsNames.sum_index_column, ColumnsNames.sum_rank_column],
                       ascending=[False, True], inplace=True)
        df = df[[ColumnsNames.features_indices_column,
                 'feature_x',
                 ColumnsNames.sum_index_column,
                 ColumnsNames.sum_rank_column]]

        df = df.rename(columns={'feature_x': ColumnsNames.features_name_column,
                                ColumnsNames.sum_index_column: f'{ColumnsNames.sum_index_column}_sfs',
                                ColumnsNames.sum_rank_column: f'{ColumnsNames.sum_rank_column}_sfs'})
        return df


def combine_multivariate_feature_selection(dfs_list):
    for df in dfs_list:
        df.sort_values(by=[ColumnsNames.features_indices_column], ascending=[True], inplace=True)
    cols = [ColumnsNames.features_indices_column, ColumnsNames.features_name_column]
    df_multi: pd.DataFrame = pd.concat([d.set_index(cols) for d in dfs_list], axis=1).reset_index()
    df_multi[ColumnsNames.sum_index_column] = df_multi.filter(regex=f"{ColumnsNames.sum_index_column}_.*") \
        .sum(axis=1)
    df_multi[ColumnsNames.sum_rank_column] = df_multi.filter(regex=f"{ColumnsNames.sum_rank_column}_.*") \
        .sum(axis=1)
    df_multi.sort_values(by=[ColumnsNames.sum_index_column, ColumnsNames.sum_rank_column],
                         ascending=[False, True], inplace=True)
    return df_multi


def get_multivariate_feature_indices(data_df, label_df, features_indices, features_names, top_multivariate,
                                     multivariate_methods_dict, cross_validation, groups):
    """

    :param data_df:
    :param label_df:
    :param features_indices:
    :param features_names:
    :param top_multivariate:
    :param multivariate_methods_dict:
    :param cross_validation:
    :param groups:
    :return:
    """
    multivariate_features_ranks = []
    for method_name, params in multivariate_methods_dict.items():
        params = params + [features_indices, features_names]
        multivariate_method_class = get_multivariate_class(method_name, params)
        multivariate_features_ranks.append(
            multivariate_method_class.run_method(data_df, label_df, cross_validation, groups))
    multi_df = combine_multivariate_feature_selection(multivariate_features_ranks)
    select_multi_df = selected_rank_features(multi_df, top_multivariate, is_min_top_features=True)

    return select_multi_df[ColumnsNames.features_indices_column].to_numpy()


# endregion

# region custom selection

def custom_feature_selection(data_df, label_df, cross_validation, groups, univariate_methods_list, min_to_select,
                             top_univariate, top_multivariate, multivariate_methods_dict):
    chosen_variance_features_indices, chosen_variance_features_names = get_variance_features(data_df)
    variance_data_df = data_df.iloc[:, chosen_variance_features_indices]

    selected_univariate_indices = get_univariate_feature_indices(variance_data_df, label_df,
                                                                 chosen_variance_features_indices,
                                                                 chosen_variance_features_names,
                                                                 univariate_methods_list, min_to_select,
                                                                 is_min_top_features=True,
                                                                 top_univariate=top_univariate)

    univariate_features_df = data_df.iloc[:, selected_univariate_indices]

    multivariate_methods_dict = {k: [min_to_select] + v for k, v in multivariate_methods_dict.items()}
    multivariate_features_indices = get_multivariate_feature_indices(univariate_features_df, label_df,
                                                                     selected_univariate_indices,
                                                                     univariate_features_df.columns,
                                                                     top_multivariate, multivariate_methods_dict,
                                                                     cross_validation, groups)

    return data_df.iloc[:, multivariate_features_indices].columns


# endregion

# region feature importance combinations

def get_features_rank(fs_ls: list, min_to_select: int = 10, weighting: bool = False) -> pd.DataFrame:
    """
    create a data-frame with original indices, features name and relative importance that represents the relation
    strength to the target and combining the corresponding feature importance, selection indicator, and rank
    :param weighting:
    :param fs_ls: list of features importance from methods
    :param min_to_select: minimum features to select in each method
    :return: df_select_rank: data-frame which is sorted first in descending order of the sum number of methods
        that selected the feature and second in ascending order of their sum of ranks
    """
    df: pd.DataFrame = fs_ls[0]
    df_select_rank = df.drop([ColumnsNames.scoring_column], axis=1)
    df_select_rank[ColumnsNames.sum_index_column] = 0
    df_select_rank[ColumnsNames.sum_rank_column] = 0

    if weighting:
        weight = 1 / len(fs_ls)
    else:
        weight = 1

    for fs in fs_ls:
        df_ordered = fs.sort_values(by=[ColumnsNames.scoring_column], ascending=False, ignore_index=True)
        sum_score = df_ordered[ColumnsNames.scoring_column].sum()
        # select features with 95% of the accumulated score
        first_threshold = 0.95 * sum_score
        accum_df = pd.DataFrame(accumulate(df_ordered['score']))
        index1 = accum_df.index
        condition1 = accum_df < first_threshold
        condition1_indices = index1[condition1.iloc[:, 0]]
        # make sure selected features are with at least 1% of total accumulated score
        second_threshold = 0.01 * sum_score
        index2 = df_ordered[ColumnsNames.scoring_column].index
        condition2 = df_ordered[ColumnsNames.scoring_column] > second_threshold
        condition2_indices = index2[condition2]
        # intersection of two conditions
        condition_indices = condition1_indices.intersection(condition2_indices)
        condition = np.logical_and(np.array(condition1), np.array(pd.DataFrame(condition2)))
        df_ordered['rank'] = 1 + np.arange(len(condition2))
        # sanity check - minimum of features selected
        if len(condition_indices) >= min_to_select:
            df_ordered['ind'] = pd.DataFrame(condition)
        else:
            condition = df_ordered['rank'] <= min_to_select
            df_ordered['ind'] = pd.DataFrame(condition)

        df_ordered = df_ordered.sort_values(by=[ColumnsNames.features_indices_column],
                                            ascending=True, ignore_index=True)

        df_select_rank[ColumnsNames.sum_index_column] += weight * df_ordered['ind']
        df_select_rank[ColumnsNames.sum_rank_column] += weight * df_ordered['rank']

    df_select_rank.sort_values(by=[ColumnsNames.sum_index_column, ColumnsNames.sum_rank_column],
                               ascending=[False, True], inplace=True, ignore_index=True)

    return df_select_rank


def selected_rank_features(df_select_rank: pd.DataFrame, top_features: int = 50,
                           is_min_top_features: bool = True) -> pd.DataFrame:
    """
    select the minimum between input number and the most influential features (at least in one method this feature was
    selected)
    :param df_select_rank: data-frame which is sorted first in descending order of the sum number of methods
        that selected the feature and second in ascending order of their sum of ranks
    :param top_features: int that determine the number of first row of df_select_rank to return
    :param is_min_top_features: str that determines whether to select minimum or maximum top_features
    :return: data-frame with number rows of data-frame df_select_rank.
    """
    index_importance = df_select_rank[ColumnsNames.sum_index_column] >= 1
    if is_min_top_features:
        number = np.minimum(top_features, np.sum(index_importance))
    else:
        number = np.maximum(top_features, np.sum(index_importance))
    return df_select_rank[[ColumnsNames.features_indices_column, ColumnsNames.features_name_column]].head(number)


# endregion


if __name__ == '__main__':
    lab_results_df = pd.read_csv(r"C:\Developments\Projects\Placebo\data\Preprocessed\numeric_lab_results.csv",
                                 index_col=['Id', 'Date'], parse_dates=['Date'])
    lab_results_df.fillna(1, inplace=True)
    data2_df = lab_results_df
    groups2 = data2_df.index.get_level_values(0)
    label2_df = pd.DataFrame([1] * (lab_results_df.shape[0] - 1000) + [0] * 1000)
    skf = StratifiedGroupKFold(n_splits=5)
    x = custom_feature_selection(data2_df, label2_df, skf, groups2,
                                 univariate_methods_list=[f_classif, mutual_info_classif],
                                 min_to_select=10, top_univariate=50, top_multivariate=20,
                                 multivariate_methods_dict={"sfs": [], "decision_tree": []})
    print(1)
