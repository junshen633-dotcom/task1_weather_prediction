"""实验扩展：加权 Bootstrap；保留原 CART 的 Gini 与叶概率实现。"""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from models import CARTClassifier, RandomForestClassifier, check_X
from preprocessing import clean, NUMERIC


def _weighted_tree(payload):
    X, y, classes, params, seed, probability = payload
    indices = np.random.default_rng(seed).choice(len(X), size=len(X), replace=True, p=probability)
    return CARTClassifier(**params, random_state=seed).fit(X[indices], y[indices], classes=classes)


class WeightedBootstrapForest(RandomForestClassifier):
    """权重只控制抽样频率，不重复用于 Gini；不是加权损失 CART。"""
    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            return super().fit(X, y)
        X, y = check_X(X), np.asarray(y)
        w = np.asarray(sample_weight, dtype=float)
        if y.ndim != 1 or len(y) != len(X) or not len(y):
            raise ValueError('训练样本与标签不匹配')
        if w.shape != (len(X),) or not np.isfinite(w).all() or (w < 0).any() or w.max() <= 0:
            raise ValueError('权重必须为等长、有限、非负且至少一项为正的一维数组')
        if self.n_estimators < 1 or self.n_jobs < 1:
            raise ValueError('树数量与进程数必须 >= 1')
        # 全等权重走原算法，保证无降权对照逐位复现。
        if np.all(w == w[0]):
            return super().fit(X, y)
        w = w / w.max()
        probability = w / w.sum()
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        params = dict(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                      max_features=self.max_features)
        seeds = np.random.SeedSequence(self.random_state).generate_state(self.n_estimators)
        tasks = [(X, y, self.classes_, params, int(s), probability) for s in seeds]
        if self.n_jobs == 1:
            self.estimators_ = list(map(_weighted_tree, tasks))
        else:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as pool:
                self.estimators_ = list(pool.map(_weighted_tree, tasks))
        self.feature_importances_ = np.mean([t.feature_importances_ for t in self.estimators_], axis=0)
        return self


def training_anomalies(frame, policy):
    """仅传入训练折；不使用标签。零尺度列不标记，以避免将常数列误判。"""
    d = clean(frame)
    if policy == 'pressure_mad3':
        values = d[['Atmospheric Pressure']]
        center = values.median()
        scale = (values-center).abs().median() * 1.4826
        low, high = center-3*scale, center+3*scale
    elif policy == 'numeric_iqr3':
        values = d[NUMERIC]
        q1, q3 = values.quantile(.25), values.quantile(.75)
        scale = q3-q1
        low, high = q1-3*scale, q3+3*scale
    else:
        raise ValueError(f'未知规则：{policy}')
    flagged = (((values < low) | (values > high)) & (scale > 0)).any(axis=1).to_numpy()
    bounds = {c: dict(lower=float(low[c]), upper=float(high[c]), active=bool(scale[c] > 0))
              for c in values.columns}
    return flagged, bounds
