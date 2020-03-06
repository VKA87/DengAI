import os

from dataset import get_data
from config import CLEANED_DATA_PATH


def clean_ndvi(df):
    cols = ['ndvi_se', 'ndvi_sw', 'ndvi_ne', 'ndvi_se']
    ixs = {}
    for col in cols:
        ixs[col] = df[col].isna()



def clean_data(df):
    return df


def save_data(df, filename):
    df.to_csv(os.path.join(CLEANED_DATA_PATH, filename))


if __name__ == '__main__':
    train_df, test_df = get_data('train'), get_data('test')

    train_df, test_df = clean_data(train_df), clean_data(test_df)
    save_data(train_df, 'clean_train.csv')
    save_data(test_df, 'clean_test.csv')
