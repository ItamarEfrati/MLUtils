import pandas as pd


def process_file(df, date_column):
    df['USUBJID'] = df['USUBJID'].apply(lambda x: x.split('_')[-1])
    df[date_column] = pd.to_datetime(df[date_column]).dt.strftime('%Y-%m-%d')
    df = df.set_index(['USUBJID', date_column])
    df.index.names = ['Id', 'Date']
    return df
