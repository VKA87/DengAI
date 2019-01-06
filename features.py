from functools import partial

import numpy as np


def rolling_mean(values, window):
    res = np.nan * values
    cumsums = np.cumsum(values)
    shifted_cumsums = 0 * values
    shifted_cumsums[window:] = cumsums[:-window]
    res = (cumsums - shifted_cumsums) / window
    return res


def rolling_mean_per_week(df, col, weeks):
    values = df[col]
    return rolling_mean(values, weeks)


def generate_features(df, functions):
    for i, f in enumerate(functions):
        df['feat:' + str(i)] = f(df)
    return df


def get_features(df):
    return df

FEATURES = [
    partial(rolling_mean_per_week, col='ndvi_se', weeks=3),
    partial(rolling_mean_per_week, col='ndvi_ne', weeks=3),
    partial(rolling_mean_per_week, col='ndvi_nw', weeks=3),
    partial(rolling_mean_per_week, col='ndvi_sw', weeks=3),
    partial(rolling_mean_per_week, col='precipitation_amt_mm', weeks=3),
    ]

if __name__ == '__main__':
    from dataset import get_data
    df = get_data('train')
    df = df.fillna(0)
    df = generate_features(df, FEATURES)
    df = get_features(df)
