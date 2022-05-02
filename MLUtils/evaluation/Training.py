import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, roc_curve
from sklearn.model_selection import BaseCrossValidator, GridSearchCV

from MLUtils.Visualization import plot_roc_curve

LOGISTIC_REGRESSION_PARAMS = {
    "solver": ['newton-cg', 'lbfgs', 'liblinear'],
    "penalty": ['none', 'l1', 'l2', 'elasticnet'],
    "C": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': [100, 200, 500, 1000],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 6, 8, 10, 12],
    'criterion': ['gini', 'entropy']
}


def split_train_test(df: pd.DataFrame, groups: pd.Series, label_columns: str, splitter: BaseCrossValidator):
    """
    split a dataframe into train and test splits
    :param df: the dataframe to split
    :param groups: the groups of each row. Can be None
    :param label_columns: the column of the label
    :param splitter: sklearn spliter implementing BaseCrossValidator
    :return: train test splits and train groups
    """
    train_inx, test_inx = next(splitter.split(df, groups=groups))
    train_groups = list(df.drop(columns=[label_columns]).iloc[train_inx].index.get_level_values(0))
    X_train = df.drop(columns=[label_columns]).iloc[train_inx].values
    X_test = df.drop(columns=[label_columns]).iloc[test_inx].values

    y_train = df[label_columns].iloc[train_inx].values
    y_test = df[label_columns].iloc[test_inx].values

    return X_train, X_test, y_train, y_test, train_groups


def eval_grid_search(estimator,
                     grid_params: dict,
                     X_train: np.array,
                     X_test: np.array,
                     y_train: np.array,
                     y_test: np.array,
                     is_binary: bool,
                     spliter: BaseCrossValidator):
    """
    Running a full evaluation of the data. Training a classifier using gridsearch with grid parameters. Evaluating the
    test data using the best grid params and plot the classification report of the train and test data
    :param estimator: sklearn estimator to be used for the gridsearch
    :param grid_params: the parameters to test
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param is_binary:  indicates if the prediction is binary or not for roc curve plot
    :param spliter: splitter: sklearn spliter implementing BaseCrossValidator
    :return:
    """
    grid_search = GridSearchCV(estimator, grid_params, cv=spliter, n_jobs=-1, verbose=4)
    grid_search.fit(X_train, y_train)

    print("Best params")
    print(grid_search.best_params_)
    print('-' * 50)

    best_estimator = grid_search.best_estimator_

    y_pred = best_estimator.predict(X_train)

    print("Train classification report")
    print(classification_report(y_train, y_pred))
    print('-' * 50)

    y_pred = best_estimator.predict(X_test)

    print("Test classification report")
    print(classification_report(y_test, y_pred))
    print('-' * 50)

    if is_binary:
        y_pred_proba = best_estimator.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plot_roc_curve(fpr, tpr)

    return grid_search
