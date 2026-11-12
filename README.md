# COMP5434 Task 1：基于 NumPy 的天气类型分类

本项目用 NumPy 与 Python 标准库自行实现 CART 决策树、随机森林和加权 Bootstrap 森林，完成天气类型四分类，覆盖数据检查、分层交叉验证、参数搜索、库模型对照、全量训练、模型保存与提交文件生成。核心算法位于 `models.py` 与 `weighted_forest.py`；scikit-learn 仅用于评价、数据划分和明确标注的对照模型。

## 1. 目录结构

- `models.py`：NumPy 自实现 CART 决策树与随机森林，含最优分裂、叶节点概率、Bootstrap、随机特征子集和概率集成。
- `weighted_forest.py`：加权 Bootstrap 森林 `WeightedBootstrapForest`，以及无监督异常标记规则。
- `preprocessing.py`：字段校验、固定云量修复、训练折中位数填补与独热编码。
- `run.py`：统一命令行入口（audit / experiment / train / predict / all）。
- `data/`：训练与测试数据 `train.csv`、`test.csv`、`sample_submission.csv`。
- `experiments/improve_final/`：最终方案的提交文件与元数据。
- `tests/`：核心算法与可复现性单元测试。
- `requirements.txt`、`requirements-lock.txt`：依赖范围与实际使用版本。
- `RESULTS.md`、`DIAGNOSIS.md`：实验结果与准确率瓶颈诊断。

## 2. 环境配置

在项目根目录打开 Anaconda Prompt，执行：

```bat
conda create -n weather_task1 python=3.12 -y
conda activate weather_task1
python -m pip install -r requirements-lock.txt
```

若锁定版本在当前平台不可用，可改用 `python -m pip install -r requirements.txt`，此时应重新训练以保证结果一致。整个流程仅需 CPU，不需要 CUDA。

## 3. 一条命令完整运行

```bat
python run.py all --data data --out my_outputs --jobs 2
```

执行顺序：数据检查 → 80% 开发集内三折参数搜索 → 20% 留出集对照评价 → 全量训练自实现森林 → 保存模型 → 从保存的模型重新预测测试集。

固定 seed=42；搜索阶段每组参数使用较少的树，最终模型树数由 `--trees` 指定。机器负载较高时可用 `--jobs 1`。

快速验证流程（输出分数不用于正式结果）：

```bat
python run.py all --out smoke_outputs --search-trees 5 --trees 10 --jobs 1
```

## 4. 分步运行

```bat
python run.py audit --out my_outputs
python run.py experiment --out my_outputs --folds 3 --search-trees 40 --trees 200 --jobs 2
python run.py train --out my_outputs --jobs 2
python run.py predict --model my_outputs/model.pkl --out regenerated
python -m unittest discover -s tests -v
```

`experiment` 会保存开发集交叉验证选出的 `best_params.json`；`train` 若在输出目录发现该文件，会使用其中的完整参数，命令行 `--trees` 和 `--seed` 不再覆盖它。模型使用 pickle 保存，跨 Python/NumPy 版本建议重新训练。

## 5. 数据与标签

- 共 10 个特征：7 个数值特征与 3 个类别特征，`id` 不作为模型输入。
- 数值特征：Temperature、Humidity、Wind Speed、Precipitation (%)、Atmospheric Pressure、UV Index、Visibility (km)。
- 类别特征：Cloud Cover、Season、Location。
- 标签编码：1=Sunny、2=Cloudy、3=Rainy、4=Snowy。
- 类别字符串去除首尾空格并转小写；Cloud Cover 固定修复 `partly 2 → partly cloudy`、`2 → cloudy`。
- 中位数填补、类别词表、统计阈值均只在训练折学习，验证与测试集沿用训练折记录，避免数据泄漏。

数据来源于 Kaggle 数据集 Weather Type Classification，为合成练习数据，其中刻意加入了异常值，不代表真实气象观测。

## 6. 算法要点

CART 节点采用 Gini 不纯度 `1 - sum(p_c ** 2)`。对每个特征稳定排序，只考察相邻不同取值之间的阈值，借助类别指示矩阵的前缀和一次性计算各候选阈值的左右类别计数，选择加权子节点不纯度最小的分裂。叶节点保存各类别比例。

停止条件包括：节点纯、达到最大深度、样本数不足以产生两个合法叶节点，或不存在带来正增益的分裂。随机森林对 N 条记录有放回抽样 N 次，每个节点优先考察 sqrt(d) 个随机特征，最后平均所有树的类别概率并取 argmax。各树随机种子由 NumPy SeedSequence 生成，多进程训练通过 ProcessPoolExecutor 实现。

加权 Bootstrap 森林以抽样概率 `p_i = w_i / sum(w_i)` 进行有放回抽样，权重只影响抽样频率，不重复作用于 Gini 计算或叶节点概率；等权情形与原随机森林完全一致。

## 7. 最终方案

最终模型在加权森林基础上加入分层异常处理与 UV 特征：

- 气压异常样本（约 7%，标签近似随机）通过 MAD 规则识别并降权至 0.25。
- 其他数值特征异常但气压正常的样本采用 winsorize 缩尾修复，保留其有效标签信息。
- 构造 `uv_vis`（UV 指数 × 能见度）与 `uv_precip`（UV 指数 × 降水）特征，用于区分高降水区域中的 Cloudy 与 Rainy。

最终配置为 300 棵树、max_depth=12、min_samples_leaf=5、max_features=sqrt，详见 `experiments/improve_final/metadata.json`。

## 8. 结果与结论

多个留出划分下平均 Macro-F1 约 0.917–0.919，最好单次约 0.925，具体数值见 `RESULTS.md`。准确率无法继续提升的根本原因是数据中约 7% 的样本标签与特征在统计上相互独立，属于数据生成机制决定的不可约误差，完整证据链与数学推导见 `DIAGNOSIS.md`。
