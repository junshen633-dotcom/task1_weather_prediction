"""字段校验、云量修复与仅在训练折拟合的编码器。"""
import numpy as np
import pandas as pd

NUMERIC = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)',
           'Atmospheric Pressure', 'UV Index', 'Visibility (km)']
CATEGORICAL = ['Cloud Cover', 'Season', 'Location']
FEATURES = NUMERIC + CATEGORICAL
LABEL_NAMES = {1:'Sunny', 2:'Cloudy', 3:'Rainy', 4:'Snowy'}


def clean(frame):
    missing = set(FEATURES)-set(frame.columns)
    if missing:
        raise ValueError(f'缺少字段: {sorted(missing)}')
    d = frame[FEATURES].copy()
    for col in NUMERIC:
        # 非数值、inf 作为缺失处理，在训练折计算中位数后填补。
        d[col] = pd.to_numeric(d[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    for col in CATEGORICAL:
        d[col] = d[col].astype('string').str.strip().str.lower().fillna('__missing__')
    d['Cloud Cover'] = d['Cloud Cover'].replace({'partly 2':'partly cloudy', '2':'cloudy'})
    return d


class WeatherPreprocessor:
    def fit(self, frame):
        d = clean(frame)
        self.medians_ = d[NUMERIC].median().fillna(0)
        self.categories_ = {c: sorted(d[c].unique()) for c in CATEGORICAL}
        self.feature_names_ = NUMERIC + [f'{c}={v}' for c in CATEGORICAL for v in self.categories_[c]]
        return self

    def transform(self, frame):
        d = clean(frame)
        arrays = [d[NUMERIC].fillna(self.medians_).to_numpy(dtype=float)]
        for col in CATEGORICAL:
            # 未见过的类别编码为全 0，维度保持不变。
            arrays.append(np.column_stack([(d[col] == v).to_numpy(dtype=float)
                                          for v in self.categories_[col]]))
        return np.column_stack(arrays)

    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)


def load_data(directory):
    from pathlib import Path
    root = Path(directory)
    train, test, sample = [pd.read_csv(root/name) for name in
                           ['train.csv', 'test.csv', 'sample_submission.csv']]
    for name, frame in [('train', train), ('test', test)]:
        clean(frame)
        if 'id' not in frame or frame.id.isna().any() or not frame.id.is_unique:
            raise ValueError(f'{name} 的 id 必须非空且唯一')
    if 'label' not in train or not train.label.isin(LABEL_NAMES).all():
        raise ValueError('训练标签必须为整数 1/2/3/4')
    if set(train.label) != set(LABEL_NAMES):
        raise ValueError('训练集必须含四类')
    if list(sample.columns) != ['id', 'label'] or not sample.id.equals(test.id):
        raise ValueError('sample_submission 必须为 id,label，且 id 顺序与 test 一致')
    return train, test, sample


def audit(train, test):
    result = {}
    for name, frame in [('train', train), ('test', test)]:
        d = clean(frame)
        result[name] = dict(rows=len(frame), missing_raw=frame.isna().sum().to_dict(),
            missing_numeric_after_coercion=d[NUMERIC].isna().sum().to_dict(),
            duplicate_features=int(d.duplicated().sum()),
            categories_raw={c: frame[c].value_counts(dropna=False).to_dict() for c in CATEGORICAL},
            numeric_ranges={c: {'min':float(d[c].min()), 'max':float(d[c].max())} for c in NUMERIC},
            humidity_over_100=int((d.Humidity > 100).sum()),
            precipitation_over_100=int((d['Precipitation (%)'] > 100).sum()))
    result['class_counts'] = train.label.value_counts().sort_index().to_dict()
    result['policy'] = '固定云量名称映射；排除 id；保留异常数值；训练折中位数填缺；训练折独热编码'
    return result
