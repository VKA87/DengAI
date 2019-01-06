import os

import pandas as pd

from config import DATA_PATH


def get_data(fold):
    filename = 'dengue_features_{}.csv'.format(fold)
    filepath = os.path.join(DATA_PATH, filename)
    df = pd.read_csv(filepath, parse_dates=['week_start_date'])
    if fold == 'train':
        targetpath = os.path.join(DATA_PATH, 'dengue_labels_train.csv')
        targets = pd.read_csv(targetpath)
        df = df.merge(targets)
    return df
