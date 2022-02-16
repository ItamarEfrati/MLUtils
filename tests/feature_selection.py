import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedGroupKFold

from evaluation.FeatureSelection import custom_feature_selection

if __name__ == '__main__':
    lab_results_df = pd.read_csv(r"C:\Developments\Projects\Placebo\data\Preprocessed\numeric_lab_results.csv",
                                 index_col=['Id', 'Date'], parse_dates=['Date'])
    lab_results_df.fillna(1, inplace=True)
    data2_df = lab_results_df
    groups2 = data2_df.index.get_level_values(0)
    label2_df = pd.DataFrame([1] * (lab_results_df.shape[0] - 1000) + [0] * 1000)
    skf = StratifiedGroupKFold(n_splits=3)
    results_x = custom_feature_selection(data2_df, label2_df, skf, groups2,
                                         univariate_methods_list=[f_classif, mutual_info_classif],
                                         min_to_select=10, top_univariate=50, top_multivariate=20,
                                         multivariate_methods_dict={"sfs": [], "decision_tree": []})
    print(1)