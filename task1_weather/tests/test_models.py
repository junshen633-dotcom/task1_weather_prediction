"""检查算法行为、训练折隔离和跨进程复现性。"""
import unittest
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from models import CARTClassifier, RandomForestClassifier
from preprocessing import WeatherPreprocessor, NUMERIC, CATEGORICAL


class ModelTests(unittest.TestCase):
    def test_exact_cart_reference(self):
        # 连续随机数据避免大量同分阈值；全部特征参与搜索。
        rng = np.random.default_rng(9)
        X = rng.normal(size=(200,4))
        y = np.where(X[:,0] > .2, 4, np.where(X[:,1] > -.4, 2, 1))
        a = CARTClassifier(max_depth=5,min_samples_leaf=3).fit(X,y)
        b = DecisionTreeClassifier(max_depth=5,min_samples_leaf=3,random_state=42).fit(X,y)
        V = rng.normal(size=(300,4))
        np.testing.assert_array_equal(a.predict(V),b.predict(V))
        np.testing.assert_allclose(a.predict_proba(V),b.predict_proba(V))

    def test_pure_constant_and_missing_class(self):
        X = np.ones((6,2)); y = np.array([1,1,1,4,4,4])
        a = CARTClassifier().fit(X,y,classes=np.array([1,2,3,4]))
        np.testing.assert_allclose(a.predict_proba(X),np.tile([.5,0,0,.5],(6,1)))
        b = CARTClassifier().fit(X,np.ones(6))
        np.testing.assert_array_equal(b.predict(X),np.ones(6))

    def test_parallel_and_pickle(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(80,3)); y = np.where(X[:,0]>0,1,4)
        a = RandomForestClassifier(n_estimators=8,n_jobs=1).fit(X,y)
        b = RandomForestClassifier(n_estimators=8,n_jobs=2).fit(X,y)
        np.testing.assert_array_equal(a.predict_proba(X),b.predict_proba(X))
        np.testing.assert_array_equal(a.predict(X),pickle.loads(pickle.dumps(a)).predict(X))
        np.testing.assert_allclose(a.predict_proba(X).sum(axis=1),1)

    def test_preprocessing_isolation(self):
        d = pd.DataFrame({c:[1.,3.] for c in NUMERIC})
        for c in CATEGORICAL:
            d[c] = ['x','y']
        d['Cloud Cover'] = ['partly 2','2']
        p = WeatherPreprocessor().fit(d)
        v = d.copy(); v.loc[0,'Temperature'] = np.nan
        v['Cloud Cover'] = ['partly cloudy','cloudy']
        np.testing.assert_array_equal(p.transform(d)[:,7:],p.transform(v)[:,7:])
        self.assertEqual(p.transform(v)[0,0],2.)
        v['Location'] = 'unknown'
        self.assertEqual(p.transform(v).shape,p.transform(d).shape)
        self.assertNotIn('unknown',p.categories_['Location'])

    def test_reject_nan(self):
        with self.assertRaises(ValueError):
            CARTClassifier().fit([[np.nan]],[1])


if __name__ == '__main__':
    unittest.main()
