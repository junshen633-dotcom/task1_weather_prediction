# COMP5434 Task 1：NumPy 天气分类项目

本项目自行实现 CART 决策树与随机森林，提供数据检查、分层验证、参数搜索、库模型对照、全量训练、模型保存及 Kaggle CSV 生成。核心算法位于 `models.py`，只使用 NumPy 和 Python 标准库；scikit-learn 用于评价、数据划分和明确标注的对照模型。

## 1. 文件结构

- `models.py`：CART 最优分裂、叶节点概率、Bootstrap、随机特征子集和概率集成。
- `preprocessing.py`：字段校验、固定云量修复、训练折中位数与独热编码。
- `run.py`：统一命令行入口。
- `tests/test_models.py`：算法与复现性测试。
- `data/`：本次用户提供的 train.csv、test.csv、sample_submission.csv。
- `outputs/`：本次实际运行的结果、配置、留出集预测、混淆矩阵和最终模型。
- `requirements.txt`：允许的依赖范围。
- `requirements-lock.txt`：本次运行实际使用版本。
- `RESULTS.md`：本次结果和解释。

原数据仅供本课程任务使用，请遵守课程与比赛数据使用规则。项目未自动连接 Kaggle，也未提交预测。

## 2. Windows + Anaconda 安装

解压后进入 `task1_weather` 文件夹，在 Anaconda Prompt 执行：

```bat
conda create -n weather_task1 python=3.12 -y
conda activate weather_task1
python -m pip install -r requirements-lock.txt
```

若锁定版本在你的平台不可用，可执行 `python -m pip install -r requirements.txt`；这可能使库模型或序列化兼容性变化，应重新训练。Linux/macOS 也可以使用 venv。仅需 CPU，不需要 CUDA。

## 3. 一条命令完整运行

```bat
python run.py all --data data --out my_outputs --jobs 2
```

顺序：数据审查 → 80% 开发集内三折搜索 → 20% 留出集对照评价 → 全量训练自实现森林 → 保存模型 → 从保存的模型重新预测测试集。

默认参数：固定 seed=42，4 组参数，每组 3 折、每折 40 棵树；最终与对照森林均为 200 棵树。Windows 下已采用主入口保护和顶层 worker，支持多进程。机器负载较高可用 `--jobs 1`。

快速验证完整流程（分数不应当作正式结果）：

```bat
python run.py all --out smoke_outputs --search-trees 5 --trees 10 --jobs 1
```

本次正式运行命令：

```text
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 run.py all --jobs 2 --out outputs
```

Windows 若要同样限制库底层线程，可先执行：

```bat
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
python run.py all --jobs 2 --out my_outputs
```

命令会覆盖指定输出目录中的同名结果。为保留已交付的结果，请使用 `my_outputs` 等新目录。

## 4. 分步运行

```bat
python run.py audit --out my_outputs
python run.py experiment --out my_outputs --folds 3 --search-trees 40 --trees 200 --jobs 2
python run.py train --out my_outputs --jobs 2
python run.py predict --model my_outputs/model.pkl --out regenerated
python -m unittest discover -s tests -v
```

`experiment` 保存开发集 CV 选出的 `best_params.json`。`train` 若在输出目录发现它，会使用该文件的完整参数（包括树数与种子）；此时命令行 `--trees` 和 `--seed` 不覆盖它。若不存在，则使用默认深度 12、叶节点最小样本数 2，以及命令行树数/种子。需要新搜索配置时请重新执行 experiment 或使用新输出目录。

`predict` 当前仍要求 `--data` 目录存在三份 CSV，以便执行统一校验；它不会重新训练。模型使用 pickle 保存，只加载你自己生成且信任的模型文件。跨 Python/NumPy 版本建议重新训练。

## 5. 数据处理及标签格式

- 输入为 7 个数值与 3 个类别字段，明确排除 id。
- 去除类别字符串首尾空格并转小写。
- 固定修复 `partly 2 → partly cloudy`、`2 → cloudy`，只作用于 Cloud Cover。
- 不删除超过 100 的湿度/降水数值，不裁剪异常值，不使用测试标签。
- 数值中位数和类别词表仅从当前训练折学习；验证/测试未知类别为全 0，缺失类别使用 `__missing__`。
- 原始数据没有缺失值，但代码支持数值强制转换、inf 和缺失值填补；某数值列训练部分全缺失时填 0。
- 数值不标准化；逻辑回归对照另行使用 StandardScaler。

PDF/训练标签为 1=Sunny、2=Cloudy、3=Rainy、4=Snowy，但样例提交是英文 Rainy。本项目总是输出两个候选格式：

- `submission_english.csv`：英文天气名。
- `submission_integer.csv`：整数 1–4。
- `submission.csv`：按 `--label-format auto|english|integer` 选择；auto 根据样例列判断，本数据自动选英文。

提交前请以 Kaggle Evaluation 规则核实。这里已检查行数、无缺失和 ID 顺序，但没有在 Kaggle 验证接受情况，不要把两种编码盲目都提交。

## 6. 算法细节

CART 节点使用 Gini：`1 - sum(p_c ** 2)`。逐特征稳定排序，只考虑相邻不同值之间的阈值；用类别指示矩阵的前缀和同时计算候选阈值的左右类别数量。选择加权子节点不纯度最小的阈值。叶节点保存各类比例。

停止条件：纯节点、达到最大深度、样本数不足以生成两个合法叶节点，或无有效正增益分裂。`min_impurity_decrease` 使用当前节点样本占全树样本的比例加权。森林每棵树对 N 条记录有放回抽取 N 次，每个节点优先搜索 sqrt(d) 个随机特征；若没有有效分裂则继续检查剩余特征。最后平均所有树的类别概率并 argmax。

每棵树的 seed 由 NumPy SeedSequence 产生，与进程数和任务调度顺序无关。多进程训练通过 ProcessPoolExecutor 实现，当前会向 worker 传输数据，适合本数据规模；并未实现共享内存或分布式训练。

与 sklearn 不要求逐条预测一致：随机数生成器、同分阈值选择和实现细节可能不同。测试中的简单连续数据用来核验相同 CART 决策规则。完整数据主要对比 Macro-F1 和混淆矩阵。

## 7. 实验协议与结果解释

固定 7,920 条开发集、1,980 条留出集。开发集内三折搜索：max_depth ∈ {8,12}，min_samples_leaf ∈ {2,5}。只按开发集平均 Macro-F1 选参数，未按留出集分数选择最终模型；搜索使用 40 棵树，最终增为 200 棵，因此 CV 分数不是最终 200 棵模型的 CV 估计。

对照包括 NumPy CART、sklearn CART、NumPy 随机森林、sklearn 随机森林、逻辑回归和直方图梯度提升。比较表保存在 comparison.csv，行是真实类别、列是预测类别的混淆矩阵有 CSV 和 PNG 两种形式。每类 precision/recall/F1 在各模型 report.json 中。

该数据此前已用于探索性三折模型比较，因此这里留出集仅是本次搜索流程中未参与调参的集合，不能称为整个研究从未查看过的盲测集。Kaggle 隐藏测试集才是最终外部评价。不要持续根据留出集调参后仍声称它提供无偏泛化估计。

计时采用 perf_counter：comparison.csv 的训练时间仅为模型 fit；最终 offline_seconds 包括全量预处理与训练，不包含写文件；test_inference_seconds 包括测试预处理和概率预测，不包含加载模型及 CSV 写入。森林无 epoch，元数据明确记录“不适用”。硬件/包版本和输入 SHA256 见 training_metadata.json。

feature_importances.csv 是基于 Gini 下降的训练内重要性，可能偏向高基数特征，不能直接当因果解释。未实现 SHAP、类别配额约束、概率融合或异常值消融，避免把尚未完成的实验描述成成果。

## 8. 测试与可复现交付

五项测试覆盖：CART 与参考实现的已知分裂行为、纯类别/常数特征、Bootstrap 缺失类别时概率维度、串行并行一致、pickle 往返、云量修复、未知类别与训练中位数隔离，以及非有限输入拒绝。

最终训练会加载磁盘模型重新生成预测，并与内存模型比较。正式报告请区分本项目提供的实现与团队后续修改，理解核心算法后按课程要求说明工具使用和个人贡献。

## 9. 2026-10-22 后续开发集实验

上文结果与“不含异常消融”等描述对应最初交付版本；最新状态见上一级 `PROJECT_CONTEXT.md`。后续实验不会覆盖 `outputs/`，不会自动提交 Kaggle。

- `improve_macro_f1.py`：固定原开发集三折，比较 8 个自实现森林配置、4 个训练异常降权配置和3个 CatBoost 对照；保存配置、依赖版本、源代码快照、每折训练/预测耗时、OOF 概率和完整指标。
- `weighted_forest.py`：复用原 NumPy CART，增加按 `w_i / sum(w)` 有放回抽取 N 次的 Bootstrap。抽样后仍用普通计数计算 Gini 和叶概率，不对样本再加一次权；因此应称为“加权 Bootstrap”，而不是实现了加权损失 CART。等权走原实现，保持原基线预测一致。
- 异常标记完全在当前训练折计算：气压超出 `median ± 3 × 1.4826 × MAD`，或任一数值特征超出 `[Q1−3×IQR, Q3+3×IQR]`。异常权重比较0.5和0.25，正常记录为1；验证记录全部保留且正常计分。这些统计规则不代表记录或标签无效。
- CatBoost 是第三方模型，使用固定600轮及原生类别特征，不用验证折做早停。自己的贡献是自实现森林及加权抽样扩展、训练折隔离和可复现实验，而非 CatBoost 算法。

本机使用工作区根目录 `.venv-weather`（继承现有数值计算包，新装 CatBoost 及所需缺失依赖到虚拟环境）。在 `task1_weather` 内运行：

```powershell
../.venv-weather/Scripts/python.exe improve_macro_f1.py --out experiments/improvement_new
```

脚本要求原来的 `experiments/development_cv_20261022/fold_ids.json` 和 NumPy OOF 基线文件存在，确保完全相同的划分与基线。输出目录必须不存在。跨机器复现可新建独立环境，安装新实验目录内的 `requirements-experiment-lock.txt`；不要用它覆盖原始依赖记录。

训练/预测计时不含预处理，总时间包含预处理、结果落盘和验证。所有候选都是开发集搜索成绩，选中候选的分数具有选择偏差；后续需要重复划分或嵌套验证确认稳定性。

## 10. 模型表达方式实验

`density_model.py` 新增 NumPy 自实现类条件直方图概率模型：训练折数值分箱和类别词表、加一平滑、对数域概率计算，以及用 EM 软责任度拟合的共享背景混合。背景成分是统计假设，不代表无效样本或已证实的数据生成来源。`experiment_density.py` 比较固定分箱数与固定权重森林融合；不改标签、不删除验证记录。

`experiment_interactions.py` 在原自实现森林输入中追加季节×温度、云量×降水/UV等固定交互项，比较两种分裂特征抽样数。两组实验均只使用原开发集，在划分种子42、7、123下分别进行三折验证，森林随机种子固定42；复用同折历史森林预测作配对对照。概率实现没有调用第三方分类器，原CART和森林代码保持不变。

复跑时在项目目录执行 `../.venv-weather/Scripts/python.exe experiment_density.py --out experiments/新的概率实验目录` 或 `../.venv-weather/Scripts/python.exe experiment_interactions.py --out experiments/新的交互实验目录`，输出目录必须不存在。当前结果见 `experiments/interactions_cv_20261022/REPORT.md`；最高开发集成绩仍不能当作独立测试分数。

## 11. 非树模型嵌套对照

`experiment_svm.py` 使用第三方sklearn/LIBSVM RBF SVC，外层开发集三折、训练折内部三折选C和gamma，填补/编码/缩放分别在相应训练部分拟合。自己的贡献仍为原NumPy CART/森林、隔离流程与评估设计，不宣称SVM自实现。外层划分种子42/7/123，内层选择只使用外训练数据。

运行：`../.venv-weather/Scripts/python.exe experiment_svm.py --out experiments/新的SVM目录`。结果见 `experiments/svm_nested_cv_20261022/REPORT.md`。三划分平均Macro-F1为0.907785，低于对应原森林0.913645，本轮不替换主模型。OOF决策分数不是概率；没有进行概率融合。全部为开发集研究结果。

## 12. 浅层神经网络对照

`experiment_mlp.py` 使用第三方sklearn MLPClassifier（64→32、ReLU、Adam、L2），开发集外层三折、训练折内部三折联合选择正则化和训练轮数，所有填补/编码/缩放在相应训练部分拟合。主要比较固定初始化42，另报告7/123初始化和固定等权集成，不挑最好种子。原自实现森林及作业算法贡献保持不变。

运行：`../.venv-weather/Scripts/python.exe experiment_mlp.py --out experiments/新的MLP目录`。结果见 `experiments/mlp_nested_cv_20261022/REPORT.md`。三划分平均Macro-F1：主MLP0.906229、三初始化集成0.907829、同划分原森林0.913645。本轮不替换主模型；全部为已探索开发集上的结果。

## 13. 输入区域门控与专门模型

specialist_model.py实现训练输入拟合的气压MAD/数值IQR门控、区域浅森林和先验收缩，门控不使用真实标签或错误标记。运行experiment_specialist.py --out experiments/新的专门模型目录，结果见experiments/specialist_cv_20261022/REPORT.md。主要方案平均Macro-F1为0.913165，原森林0.913645；未达到跨划分稳定提升标准，正式模型不变。

## 14. 论文六特征与森林参数消融

experiment_paper_rf.py固定六特征及论文表4参数，比较同库原参数/论文参数、全特征/六特征、独热/整数编码，并保留NumPy六特征对照。不删除任何开发训练或验证记录，填补和词表仅训练折拟合。运行experiment_paper_rf.py --out experiments/新的论文实验目录。结果见experiments/paper_six_rf_cv_20261022/REPORT.md：主要六特征论文参数整数编码方案平均Macro-F1=0.911712，低于原NumPy森林0.913645，不升级正式模型。原自实现代码保持不变，论文参数使用sklearn作第三方对照。

## 15. 删除异常与评估范围对照

experiment_deletion.py比较固定公开范围与训练折IQR1.5规则，分别考察仅删训练的全量成绩，以及同一批预测在保留/排除区域的诊断指标。验证预测始终覆盖全部7,920条，不删除原CSV或改标签。运行experiment_deletion.py --out experiments/新的删除实验目录。

结果见experiments/deletion_audit_20261023/REPORT.md：论文配置不重新训练、只筛选IQR保留区评分，Accuracy从91.16%升至97.18%；删训练后完整验证Accuracy为91.08%，未稳定改善。子集高分不得当作比赛整体成绩，正式模型不变。
