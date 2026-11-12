"""验证降权行为、原基线兼容、并行复现和训练异常规则。"""
import pickle
import unittest
import numpy as np
import pandas as pd
from models import RandomForestClassifier
from weighted_forest import WeightedBootstrapForest, training_anomalies
from preprocessing import NUMERIC, CATEGORICAL


class WeightedForestTests(unittest.TestCase):
    def test_uniform_weights_preserve_baseline(self):
        X = np.random.default_rng(4).normal(size=(100, 3))
        y = np.where(X[:, 0] > 0, 1, 4)
        a = RandomForestClassifier(n_estimators=8).fit(X, y)
        b = WeightedBootstrapForest(n_estimators=8).fit(X, y, np.ones(100))
        np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))

    def test_weighted_class_mass_parallel_and_pickle(self):
        X = np.ones((200, 2)); y = np.repeat([1, 4], 100)
        weights = np.repeat([1., .25], 100)
        a = WeightedBootstrapForest(n_estimators=40).fit(X, y, weights)
        b = WeightedBootstrapForest(n_estimators=40, n_jobs=2).fit(X, y, weights)
        # 常数特征只有叶节点，概率应接近加权抽样后的 0.8/0.2。
        np.testing.assert_allclose(a.predict_proba(X)[0], [.8, .2], atol=.025)
        np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))
        np.testing.assert_array_equal(a.predict_proba(X), pickle.loads(pickle.dumps(a)).predict_proba(X))
        z = WeightedBootstrapForest(n_estimators=3).fit(X, y, np.repeat([1., 0.], 100))
        np.testing.assert_array_equal(z.predict_proba(X)[0], [1., 0.])

    def test_invalid_weights(self):
        for weights in [[0, 0], [-1, 1], [np.nan, 1], [np.inf, 1], [1], [[1], [1]]]:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                WeightedBootstrapForest(n_estimators=1).fit([[1], [2]], [1, 4], weights)

    def test_anomaly_rule_known_training_distribution(self):
        d = pd.DataFrame({c: [1000., 1001., 1002., 1003., 1500.] for c in NUMERIC})
        for c in CATEGORICAL:
            d[c] = 'known'
        for policy in ['pressure_mad3', 'numeric_iqr3']:
            mask, bounds = training_anomalies(d, policy)
            np.testing.assert_array_equal(mask, [False, False, False, False, True])
            self.assertLess(bounds['Atmospheric Pressure']['upper'], 1500)
            constant = d.copy()
            constant[NUMERIC] = 1.
            self.assertFalse(training_anomalies(constant, policy)[0].any())


if __name__ == '__main__':
    unittest.main()
