# 本次实际运行结果

固定 seed=42；开发集 7,920 条、留出集 1,980 条；四组参数各进行三折搜索。

| 模型 | 留出 Macro-F1 | 训练秒 | 推理秒 |
|---|---:|---:|---:|
| numpy_cart | 0.900274 | 0.1856 | 0.0011 |
| sklearn_cart | 0.901272 | 0.0302 | 0.0005 |
| numpy_forest | 0.912135 | 5.4893 | 0.2484 |
| sklearn_forest | 0.912252 | 0.6750 | 0.0328 |
| logistic_regression | 0.864712 | 0.0309 | 0.0007 |
| hist_gradient_boosting | 0.909734 | 0.8707 | 0.0478 |

最终森林配置：`{"max_depth": 12, "min_samples_leaf": 5, "n_estimators": 200, "max_features": "sqrt", "random_state": 42}`。

全量训练耗时 5.975 秒，预处理加训练 6.015 秒，3,300 条测试预处理加推理 0.311 秒。

运行 CPU：AMD EPYC 9V74 80-Core Processor；训练森林使用 2 个 worker。耗时仅代表当前环境，不代表用户 Windows 电脑。

5 项单元测试通过；独立 predict 命令重新加载模型后生成的 submission.csv 与首次生成文件逐字节一致。

样例使用英文标签，submission.csv 因此选英文；同时附整数版本供核对。未提交 Kaggle，也没有 Kaggle 分数。

此前探索使用过完整训练数据；本次留出集仅在本次搜索中隔离，泛化解释详见 README。开发集搜索选择 max_depth=12、min_samples_leaf=5；选参后树数量从 40 增至 200。

自实现森林与库森林分数接近；当前结果不代表自实现算法普遍更优。
