import pandas as pd


def read_panss_scores():
    return pd.read_csv(r"data\Preprocessed\panss_scores.csv", index_col=['Id', 'Date'], parse_dates=['Date'])


def read_lab_results():
    return pd.read_csv(r"data\Preprocessed\lab_results.csv", index_col=['Id', 'Date'], parse_dates=['Date'])
