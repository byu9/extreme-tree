#!/usr/bin/env python3
import pandas as pd

filenames = ['GA', 'HI', 'OR', 'TX']

quantiles = {
    'q=0.1': 0.1,
    'q=0.2': 0.2,
    'q=0.3': 0.3,
    'q=0.4': 0.4,
    'q=0.5': 0.5,
    'q=0.6': 0.6,
    'q=0.7': 0.7,
    'q=0.8': 0.8,
    'q=0.9': 0.9,
}


def pinball_loss(df_predict, df_target):
    tau = pd.DataFrame(quantiles, index=df_target.index)
    loss = pd.DataFrame(index=df_predict.index, columns=df_predict.columns)

    mask = df_target >= df_predict
    loss[mask] = (df_target[mask] - df_predict[mask]) * tau[mask]
    loss[~mask] = (df_predict[~mask] - df_target[~mask]) * (1 - tau[~mask])

    return loss.mean(axis=None)


results = pd.DataFrame(index=filenames)

for filename in filenames:
    test = pd.read_csv(f'small_datasets/compiled_datasets/test-{filename}.csv', index_col='local')
    qr_predict = pd.read_csv(f'qr_small_datasets/{filename}.csv', index_col='local')
    gev_predict = pd.read_csv(f'gev_small_datasets/{filename}.csv', index_col='local')

    test.index = pd.to_datetime(test.index, utc=True)
    qr_predict.index = pd.to_datetime(qr_predict.index, utc=True)
    gev_predict.index = pd.to_datetime(gev_predict.index, utc=True)

    target = pd.DataFrame({label: test['value'] for label in quantiles.keys()})

    results.loc[filename, 'qr'] = pinball_loss(qr_predict, target)
    results.loc[filename, 'extreme_tree'] = pinball_loss(gev_predict, target)

results['Improved'] = (results['qr'] - results['extreme_tree']) / results['qr'] * 100
mean_improved = results['Improved'].mean()
print(results)
print(f'Mean improved {mean_improved}')
