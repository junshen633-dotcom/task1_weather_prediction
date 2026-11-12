"""纯 NumPy CART / 随机森林；不调用 sklearn 的训练或预测实现。"""
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def check_X(X, n_features=None):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
        raise ValueError('X 必须为有限数值二维矩阵，且至少含一个特征')
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError('预测特征数与训练不一致')
    return X


class CARTClassifier:
    def __init__(self, max_depth=12, min_samples_leaf=2, max_features=None,
                 min_impurity_decrease=0.0, random_state=42):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state

    def fit(self, X, y, classes=None):
        X = check_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X) or len(y) == 0:
            raise ValueError('y 需为与 X 等长的非空一维数组')
        if self.min_samples_leaf < 1 or (self.max_depth is not None and self.max_depth < 0):
            raise ValueError('叶节点最小样本数 >= 1，最大深度 >= 0')
        if self.min_impurity_decrease < 0:
            raise ValueError('最小不纯度下降不能为负')
        self.classes_ = np.unique(y) if classes is None else np.asarray(classes)
        lookup = {c: i for i, c in enumerate(self.classes_)}
        yi = np.array([lookup[c] for c in y], dtype=int)
        self.n_features_in_ = X.shape[1]
        self.n_classes_ = len(self.classes_)
        self.n_samples_ = len(X)
        if self.max_features is None:
            self.m_ = X.shape[1]
        elif self.max_features == 'sqrt':
            self.m_ = max(1, int(np.sqrt(X.shape[1])))
        else:
            self.m_ = int(self.max_features)
            if not 1 <= self.m_ <= X.shape[1]:
                raise ValueError('max_features 超出特征数范围')
        rng = np.random.default_rng(self.random_state)
        self.feature_importances_ = np.zeros(X.shape[1])

        def grow(indices, depth):
            n = len(indices)
            labels = yi[indices]
            counts = np.bincount(labels, minlength=self.n_classes_)
            probability = counts / n
            impurity = 1 - probability @ probability
            # 节点格式：(特征, 阈值, 左子树, 右子树, 类别概率)
            leaf = (-1, None, None, None, probability)
            if (np.count_nonzero(counts) == 1 or n < 2*self.min_samples_leaf
                    or (self.max_depth is not None and depth >= self.max_depth)):
                return leaf
            best_gain, best = 0.0, None
            # 若抽中的特征全为常数，继续查找，避免无有效分裂时过早停止。
            features = rng.permutation(X.shape[1])
            for k, j in enumerate(features):
                if k >= self.m_ and best is not None:
                    break
                order = np.argsort(X[indices, j], kind='stable')
                values = X[indices[order], j]
                positions = np.flatnonzero(values[:-1] < values[1:])
                nl = positions + 1
                valid = (nl >= self.min_samples_leaf) & (n-nl >= self.min_samples_leaf)
                positions, nl = positions[valid], nl[valid]
                if len(positions) == 0:
                    continue
                # 排序后累计四类计数，一次计算所有合法阈值的分裂得分。
                indicators = np.eye(self.n_classes_, dtype=np.int64)[labels[order]]
                left = np.cumsum(indicators, axis=0)[positions].astype(float)
                right = counts - left
                nr = n-nl
                child_impurity = 1 - ((left*left).sum(1)/nl +
                                      (right*right).sum(1)/nr)/n
                gains = impurity-child_impurity
                q = int(np.argmax(gains))
                if gains[q] > best_gain + 1e-14:
                    pos = positions[q]
                    threshold = values[pos] + (values[pos+1]-values[pos])/2
                    if threshold >= values[pos+1]:
                        threshold = values[pos]
                    best_gain = float(gains[q])
                    best = (int(j), float(threshold))
            # 使用相对于整棵树训练样本数加权的 impurity decrease。
            if best is None or n/self.n_samples_*best_gain < self.min_impurity_decrease:
                return leaf
            j, threshold = best
            mask = X[indices, j] <= threshold
            self.feature_importances_[j] += n/self.n_samples_*best_gain
            return (j, threshold, grow(indices[mask], depth+1),
                    grow(indices[~mask], depth+1), probability)

        self.root_ = grow(np.arange(len(X)), 0)
        total = self.feature_importances_.sum()
        if total:
            self.feature_importances_ /= total
        return self

    def predict_proba(self, X):
        X = check_X(X, self.n_features_in_)
        out = np.empty((len(X), self.n_classes_))
        def visit(node, indices):
            if not len(indices):
                return
            j, threshold, left, right, p = node
            if j == -1:
                out[indices] = p
                return
            mask = X[indices, j] <= threshold
            visit(left, indices[mask])
            visit(right, indices[~mask])
        visit(self.root_, np.arange(len(X)))
        return out

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def _fit_tree(payload):
    # 顶层函数可在 Windows spawn 模式下被子进程导入。
    X, y, classes, params, seed = payload
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(X), size=len(X))
    tree = CARTClassifier(**params, random_state=seed)
    return tree.fit(X[indices], y[indices], classes=classes)


class RandomForestClassifier:
    def __init__(self, n_estimators=200, max_depth=12, min_samples_leaf=2,
                 max_features='sqrt', random_state=42, n_jobs=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X, y = check_X(X), np.asarray(y)
        if y.ndim != 1 or len(y) != len(X) or not len(y):
            raise ValueError('训练样本与标签不匹配')
        if self.n_estimators < 1 or self.n_jobs < 1:
            raise ValueError('树数量与进程数必须 >= 1')
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        params = dict(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                      max_features=self.max_features)
        # 每棵树的随机种子独立于调度次序，串行/并行输出相同。
        seeds = np.random.SeedSequence(self.random_state).generate_state(self.n_estimators)
        tasks = [(X, y, self.classes_, params, int(seed)) for seed in seeds]
        if self.n_jobs == 1:
            self.estimators_ = list(map(_fit_tree, tasks))
        else:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as pool:
                self.estimators_ = list(pool.map(_fit_tree, tasks))
        self.feature_importances_ = np.mean([t.feature_importances_ for t in self.estimators_], axis=0)
        return self

    def predict_proba(self, X):
        X = check_X(X, self.n_features_in_)
        out = np.zeros((len(X), len(self.classes_)))
        for tree in self.estimators_:
            out += tree.predict_proba(X)
        return out/len(self.estimators_)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
