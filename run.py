"""入口：audit / experiment / train / predict / all。"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier as SklearnForest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models import CARTClassifier, RandomForestClassifier
from preprocessing import WeatherPreprocessor, load_data, audit, LABEL_NAMES


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=lambda x:
                         x.item() if isinstance(x, np.generic) else str(x)), encoding='utf-8')


def environment():
    return {'python': platform.python_version(), 'platform':platform.platform(),
            'processor':platform.processor(), 'logical_cpus':os.cpu_count(),
            'packages':{p:importlib.metadata.version(p) for p in
                        ['numpy', 'pandas', 'scikit-learn', 'matplotlib']}}


def measure(name, model, X, y, V, v, ids, out):
    start = time.perf_counter()
    model.fit(X, y)
    fit_seconds = time.perf_counter()-start
    start = time.perf_counter()
    pred = model.predict(V)
    prediction_seconds = time.perf_counter()-start
    score = f1_score(v, pred, average='macro', labels=[1,2,3,4], zero_division=0)
    cm = confusion_matrix(v, pred, labels=[1,2,3,4])
    save_json(out/f'{name}_report.json', classification_report(v, pred,
              labels=[1,2,3,4], target_names=list(LABEL_NAMES.values()),
              output_dict=True, zero_division=0))
    pd.DataFrame(cm, index=LABEL_NAMES.values(), columns=LABEL_NAMES.values()).to_csv(out/f'{name}_confusion.csv')
    pd.DataFrame({'id':ids, 'true':v, 'predicted':pred}).to_csv(out/f'{name}_predictions.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5,4))
    ax.imshow(cm, cmap='Blues')
    ax.set(xticks=range(4), yticks=range(4), xticklabels=LABEL_NAMES.values(),
           yticklabels=LABEL_NAMES.values(), xlabel='Predicted', ylabel='True', title=name)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
    fig.tight_layout()
    fig.savefig(out/f'{name}_confusion.png', dpi=160)
    plt.close(fig)
    row = dict(model=name, macro_f1=score, train_seconds=fit_seconds,
               inference_seconds=prediction_seconds)
    print(row, flush=True)
    return row


def experiment(train, args, out):
    # 留出集不参与搜索；与此前探索性三折实验有样本重叠，并非完全未见的新数据。
    dev, hold = train_test_split(np.arange(len(train)), test_size=0.2,
                                 stratify=train.label, random_state=args.seed)
    save_json(out/'split_ids.json', {'development_ids':train.iloc[dev].id.tolist(),
              'holdout_ids':train.iloc[hold].id.tolist(), 'seed':args.seed})
    data = train.iloc[dev].reset_index(drop=True)
    y = data.label.to_numpy()
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    # 每折独立拟合编码器，再复用于不同参数，避免预处理泄漏。
    folds = []
    for tr, va in cv.split(data, y):
        prep = WeatherPreprocessor()
        folds.append((prep.fit_transform(data.iloc[tr]), y[tr],
                      prep.transform(data.iloc[va]), y[va]))
    candidates = [dict(max_depth=d, min_samples_leaf=l) for d in [8,12] for l in [2,5]]
    rows = []
    for config in candidates:
        scores, seconds = [], []
        for X, yt, V, yv in folds:
            model = RandomForestClassifier(n_estimators=args.search_trees, n_jobs=args.jobs,
                                           random_state=args.seed, **config)
            start = time.perf_counter()
            model.fit(X, yt)
            scores.append(f1_score(yv, model.predict(V), average='macro'))
            seconds.append(time.perf_counter()-start)
        row = dict(**config, mean_f1=float(np.mean(scores)), std_f1=float(np.std(scores)),
                   fold_scores=scores, mean_fit_predict_seconds=float(np.mean(seconds)))
        rows.append(row)
        print('CV:', row, flush=True)
        save_json(out/'search_results.json', rows)
    # 排序稳定：若完全同分，优先候选列表中较简单的模型。
    best = max(rows, key=lambda row:row['mean_f1'])
    config = {key:best[key] for key in ['max_depth','min_samples_leaf']}
    config.update(n_estimators=args.trees, max_features='sqrt', random_state=args.seed)
    save_json(out/'best_params.json', config)
    prep = WeatherPreprocessor()
    X = prep.fit_transform(train.iloc[dev]); V = prep.transform(train.iloc[hold])
    yt, yv = train.iloc[dev].label.to_numpy(), train.iloc[hold].label.to_numpy()
    models = {
        'numpy_cart':CARTClassifier(max_depth=12, min_samples_leaf=2, random_state=args.seed),
        'sklearn_cart':DecisionTreeClassifier(max_depth=12, min_samples_leaf=2, random_state=args.seed),
        'numpy_forest':RandomForestClassifier(**config, n_jobs=args.jobs),
        'sklearn_forest':SklearnForest(**config, n_jobs=args.jobs),
        'logistic_regression':make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        'hist_gradient_boosting':HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15,
                                    l2_regularization=1, random_state=args.seed)
    }
    results = [measure(name, model, X, yt, V, yv, train.iloc[hold].id.to_numpy(), out)
               for name, model in models.items()]
    pd.DataFrame(results).to_csv(out/'comparison.csv', index=False)
    save_json(out/'experiment_protocol.json', dict(seed=args.seed, folds=args.folds,
              search_trees=args.search_trees, final_trees=args.trees, jobs=args.jobs,
              selection='仅由开发集 CV 选择自实现森林参数；留出分数不用于选择最终模型',
              caveat='原始训练数据此前已用于探索性三折模型比较，不能声称此留出集从未被探索'))


def train_final(train, test, sample, args, out):
    config_path = out/'best_params.json'
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding='utf-8'))
    else:
        config = dict(n_estimators=args.trees, max_depth=12, min_samples_leaf=2,
                      max_features='sqrt', random_state=args.seed)
    start = time.perf_counter()
    prep = WeatherPreprocessor()
    X = prep.fit_transform(train)
    fit_start = time.perf_counter()
    model = RandomForestClassifier(**config, n_jobs=args.jobs).fit(X, train.label.to_numpy())
    fit_seconds = time.perf_counter()-fit_start
    offline_seconds = time.perf_counter()-start
    artifact = dict(preprocessor=prep, model=model, label_names=LABEL_NAMES,
                    config=config, environment=environment())
    with open(out/'model.pkl','wb') as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
    pd.DataFrame({'feature':prep.feature_names_, 'importance':model.feature_importances_}).sort_values(
        'importance', ascending=False).to_csv(out/'feature_importances.csv', index=False)
    timing = predict_file(test, sample, out/'model.pkl', out, args.label_format)
    # 核实磁盘模型与内存模型一致。
    disk_predictions = pd.read_csv(out/'prediction_details.csv')['label_int'].to_numpy()
    np.testing.assert_array_equal(model.predict(prep.transform(test)), disk_predictions)
    save_json(out/'training_metadata.json', dict(params=config, train_rows=len(train), test_rows=len(test),
        jobs=args.jobs, train_seconds=fit_seconds, offline_seconds=offline_seconds, **timing,
        epoch='不适用：随机森林没有 epoch', environment=environment(),
        input_sha256={name:hashlib.sha256((Path(args.data)/name).read_bytes()).hexdigest()
                      for name in ['train.csv','test.csv','sample_submission.csv']}))


def predict_file(test, sample, model_path, out, label_format):
    # pickle 只读取自己生成且可信的文件。
    with open(model_path,'rb') as f:
        artifact = pickle.load(f)
    prep, model = artifact['preprocessor'], artifact['model']
    start = time.perf_counter()
    p = model.predict_proba(prep.transform(test))
    labels = model.classes_[p.argmax(axis=1)]
    seconds = time.perf_counter()-start
    names = [artifact['label_names'][int(c)] for c in labels]
    pd.DataFrame({'id':test.id, 'label':names}).to_csv(out/'submission_english.csv', index=False)
    pd.DataFrame({'id':test.id, 'label':labels}).to_csv(out/'submission_integer.csv', index=False)
    if label_format == 'auto':
        label_format = 'english' if sample.label.isin(LABEL_NAMES.values()).all() else 'integer'
    values = names if label_format == 'english' else labels
    sub = pd.DataFrame({'id':test.id, 'label':values})
    if len(sub) != len(test) or sub.isna().any().any() or not sub.id.equals(sample.id):
        raise ValueError('提交文件校验失败')
    sub.to_csv(out/'submission.csv', index=False)
    detail = pd.DataFrame({'id':test.id, 'label_int':labels})
    for i,c in enumerate(model.classes_):
        detail[f'p_{LABEL_NAMES[int(c)]}'] = p[:,i]
    detail.to_csv(out/'prediction_details.csv', index=False)
    print(f'生成 {len(sub)} 行 submission.csv，标签格式={label_format}', flush=True)
    return {'test_inference_seconds':seconds, 'label_format':label_format}


def main():
    parser = argparse.ArgumentParser(description='COMP5434 Task 1：NumPy 随机森林')
    parser.add_argument('command', choices=['audit','experiment','train','predict','all'])
    parser.add_argument('--data', default='data')
    parser.add_argument('--out', default='outputs')
    parser.add_argument('--model', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--search-trees', type=int, default=40)
    parser.add_argument('--trees', type=int, default=200)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--label-format', choices=['auto','english','integer'], default='auto')
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train, test, sample = load_data(args.data)
    if args.command in ['audit','all']:
        save_json(out/'data_audit.json', audit(train,test))
        print('数据检查完成', flush=True)
    if args.command in ['experiment','all']:
        experiment(train,args,out)
    if args.command in ['train','all']:
        train_final(train,test,sample,args,out)
    if args.command == 'predict':
        predict_file(test,sample,args.model or out/'model.pkl',out,args.label_format)


if __name__ == '__main__':
    main()
