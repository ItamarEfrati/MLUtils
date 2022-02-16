import pandas as pd

from PreProcess.PanssProcess import preprocess_panss

if __name__ == '__main__':
    panss_df = pd.read_excel(r"C:\Developments\Projects\Placebo\data\Listing_PANSS_MR.xlsx", index_col='Subject')
    panss_df.drop(columns='Unnamed: 41', inplace=True)
    metadata_df, panss_scores_df = preprocess_panss(panss_df)