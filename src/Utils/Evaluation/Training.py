from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV


def split_train_test(df, label_columns):
    groups = list(df.index.get_level_values(0))

    train_inx, test_inx = next(GroupShuffleSplit(test_size=.20, n_splits=2)
                               .split(df, groups=groups))
    train_groups = list(df.drop(columns=[label_columns]).iloc[train_inx].index.get_level_values(0))
    X_train = df.drop(columns=[label_columns]).iloc[train_inx].values
    X_test = df.drop(columns=[label_columns]).iloc[test_inx].values

    y_train = df[label_columns].iloc[train_inx].values
    y_test = df[label_columns].iloc[test_inx].values

    return X_train, X_test, y_train, y_test, train_groups


def eval_grid_search(estimator, grid_params, X_train, X_test, y_train, y_test, cross_validation):

    grid_search = GridSearchCV(estimator, grid_params, cv=cross_validation, n_jobs=-1, verbose=4)
    grid_search.fit(X_train, y_train)

    print("Best params")
    print(grid_search.best_params_)
    print('-' * 50)

    best_classifier = grid_search.best_estimator_

    y_pred = best_classifier.predict(X_train)

    print("Train classification report")
    print(classification_report(y_train, y_pred))
    print('-' * 50)

    y_pred = best_classifier.predict(X_test)

    print("Test classification report")
    print(classification_report(y_test, y_pred))
    print('-' * 50)

    return grid_search