from sklearn.dummy import DummyRegressor

from dataset import get_data
from features import get_features


if __name__ == '__main__':
    # get data
    train_df, test_df = get_data('train'), get_data('test')

    # get features
    train_targets = train_df.pop('total_cases')
    train_mm, test_mm = get_features(train_df), get_features(test_df)

    # fit and evaluate model
    dummy = DummyRegressor()
    dummy.fit(train_mm, train_targets)
    test_preds = dummy.predict(test_mm)
