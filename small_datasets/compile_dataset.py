import pandas as pd

filenames = ['GA', 'HI', 'OR', 'TX']

for filename in filenames:
    dataframe = pd.read_csv(f'raw/{filename}.csv', index_col='local')
    dataframe.index = pd.to_datetime(dataframe.index, utc=True)

    keep_cols = ['value', 'mo', 'hr', 'dofw', 'tmpf']
    dataframe = dataframe[keep_cols]

    dataframe['value24'] = dataframe.value.shift(24)
    dataframe['value25'] = dataframe.value.shift(25)
    dataframe['value168'] = dataframe.value.shift(168)

    max_indices = dataframe.resample('1d')['value'].idxmax()
    dataframe = dataframe.loc[max_indices]
    dataframe.dropna(inplace=True, axis='rows')

    test_split = int(len(dataframe) * 0.7)

    train = dataframe.iloc[:test_split]
    test = dataframe.iloc[test_split:]

    train.to_csv(f'compiled_datasets/train-{filename}.csv')
    test.to_csv(f'compiled_datasets/test-{filename}.csv')
