import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

filenames = ['GA', 'HI', 'OR', 'TX']

plt.figure()

for filename in filenames:
    dataframe = pd.read_csv(f'raw/{filename}.csv', index_col='local')
    dataframe.index = pd.to_datetime(dataframe.index, utc=True)
    dataframe = dataframe.resample('1h').first()

    arma_dataset = pd.DataFrame(index=dataframe.index)
    arma_dataset['value'] = dataframe['value']
    arma_dataset['value24'] = dataframe['value'].shift(24)
    arma_dataset['value25'] = dataframe['value'].shift(25)
    arma_dataset['value168'] = dataframe['value'].shift(168)
    arma_dataset.dropna(inplace=True, axis='rows')
    arma_features = arma_dataset.drop(columns='value')
    arma_target = arma_dataset['value']

    arma_model = LinearRegression()
    arma_model.fit(arma_features, arma_dataset['value'])
    arma_predict = arma_model.predict(arma_features)
    arma_residuals = arma_predict - arma_target

    residuals_dataset = pd.DataFrame(index=arma_residuals.index)
    residuals_dataset['residual'] = arma_residuals
    residuals_dataset['mo'] = dataframe['mo']
    residuals_dataset['hr'] = dataframe['hr']
    residuals_dataset['dofw'] = dataframe['dofw']
    residuals_dataset['tmpf'] = dataframe['tmpf']

    max_indices = residuals_dataset.resample('1d')['residual'].idxmax()
    residuals_dataset = residuals_dataset.loc[max_indices]
    residuals_dataset.dropna(inplace=True, axis='rows')
    residuals_dataset.sort_index(inplace=True)

    test_split = int(len(residuals_dataset) * 0.7)

    train = residuals_dataset.iloc[:test_split]
    test = residuals_dataset.iloc[test_split:]

    train.to_csv(f'compiled_datasets/train-{filename}.csv')
    test.to_csv(f'compiled_datasets/test-{filename}.csv')

    plt.clf()
    residuals_dataset['residual'].plot(title=f'{filename}')
    plt.savefig(f'compiled_datasets/{filename}.png')
