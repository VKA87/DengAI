import numpy as np

from DengAI import features


def test_rolling_mean():
    values = np.array([0., 1., 2., 3., 4., 5.])
    output = features.rolling_mean(values, 2)
    expected = np.array([0, 0.5, 1.5, 2.5, 3.5, 4.5])
    assert np.all(expected == output)
