# 项目背景

## 1. 任务目标

本项目为 COMP5434 课程 Task 1，目标是根据气象相关特征预测天气类型，属于四分类问题，类别为 Sunny、Cloudy、Rainy、Snowy。评价主指标为 Macro-F1，同时记录 Accuracy。

## 2. 数据来源

数据来自 Kaggle 数据集 Weather Type Classification。该数据集为合成练习数据，作者在数据说明中明确指出其中刻意加入了异常值，并不代表真实气象观测。这一背景对后续理解准确率瓶颈非常关键。

## 3. 数据规模与字段

- 训练集 9,900 行，四个类别各 2,475 行，类别完全平衡。
- 测试集 3,300 行。
- 共 10 个输入特征：7 个数值特征和 3 个类别特征，`id` 不进入模型。
- 数值特征：Temperature、Humidity、Wind Speed、Precipitation (%)、Atmospheric Pressure、UV Index、Visibility (km)。
- 类别特征：Cloud Cover、Season、Location。
- 标签编码：1=Sunny、2=Cloudy、3=Rainy、4=Snowy。

## 4. 方法概述

项目的核心算法全部使用 NumPy 与 Python 标准库自行实现，包括：

- CART 决策树：基于 Gini 不纯度选择最优分裂，支持深度、叶样本数和增益等停止条件。
- 随机森林：Bootstrap 有放回抽样、节点随机特征选择、概率平均集成。
- 加权 Bootstrap 森林：通过样本权重改变抽样概率，实现异常样本降权。
- 预处理：训练折内的中位数填补、类别独热编码和固定云量名称修复。

scikit-learn 仅用于评价指标、数据划分以及作为对照模型，不参与核心训练预测。

## 5. 主要结论

实验确认准确率长期停留在 91% 附近并非模型容量或调参问题，而是数据中约 7% 的样本标签与全部特征在统计上相互独立，构成不可约误差，理论整体上限约为 0.948。最终通过异常降权、分层异常处理和 UV 特征工程，将多个留出划分的平均 Macro-F1 提升到约 0.917–0.919。完整诊断见 `DIAGNOSIS.md`，结果汇总见 `RESULTS.md`。
