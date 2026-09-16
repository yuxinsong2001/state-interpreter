# Predictive GRU多随机种子稳定性结果

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：run + validate
- Origin Date：2026-09-16
- Verification Status：VERIFIED
- Version Label：gru_predictive_multiseed_result_v1

## 实验目的

第一版Predictive GRU只使用一个随机种子。本实验保持模型、训练目标、数据划分和全部超参数不变，仅使用三个随机种子重复训练，检验GRU Level是否稳定。

随机种子：

```text
20260916, 20260917, 20260918
```

## 确定性复现

seed 20260916重新训练后与上一轮单seed结果进行比较：

- 最佳epoch一致：300；
- 最佳验证MSE绝对差：0；
- checkpoint参数最大绝对差：0；
- 非零参数差异数量：0；
- 五个bearing的ρ最大绝对差：0；
- 五个bearing的next-step MSE最大绝对差：0。

因此，固定seed、数据、代码和环境时，单seed实验获得了精确复现。

## 逐seed结果

| Seed | 五bearing描述性平均ρ | 最低bearing ρ | 最高bearing ρ | Collapse |
|---:|---:|---:|---:|---:|
| 20260916 | 0.951 | 0.894 | 0.999 | 0 |
| 20260917 | 0.896 | 0.738 | 0.997 | 0 |
| 20260918 | 0.938 | 0.852 | 0.998 | 0 |

三个seed均训练300轮，最佳checkpoint均位于epoch 300。验证next-step MSE分别为0.05525、0.05563和0.05477，数值非常接近。

## 逐bearing稳定性

| Bearing | GRU ρ均值 | 样本标准差 | 最小值 | 最大值 | Level两两Pearson均值 |
|---|---:|---:|---:|---:|---:|
| Bearing1_1 | 0.9640 | 0.0004 | 0.9637 | 0.9645 | 0.9999 |
| Bearing1_2 | 0.9978 | 0.0011 | 0.9966 | 0.9989 | 0.9998 |
| Bearing1_3 | 0.9544 | 0.0106 | 0.9464 | 0.9664 | 0.9986 |
| Bearing1_4 | 0.8279 | 0.0810 | 0.7376 | 0.8941 | 0.9976 |
| Bearing1_5 | 0.8981 | 0.0563 | 0.8331 | 0.9322 | 0.9989 |

所有seed和bearing均输出非恒定Level，没有状态塌缩。

训练bearing的ρ非常稳定。Bearing1_4和Bearing1_5的ρ对初始化更敏感。虽然它们的Level曲线Pearson相关仍大于0.996，局部点的顺序变化足以明显影响Spearman相关性。轨迹总体形状稳定，不代表局部退化排序稳定。

## 与现有方法比较

| Bearing | 距离Level ρ | centered HMM ρ | GRU ρ均值±标准差 |
|---|---:|---:|---:|
| Bearing1_1 | 0.9641 | 0.9163 | 0.9640 ± 0.0004 |
| Bearing1_2 | 0.9940 | 0.8314 | 0.9978 ± 0.0011 |
| Bearing1_3 | 0.9544 | 0.9461 | 0.9544 ± 0.0106 |
| Bearing1_4 | 0.8489 | 0.9623 | 0.8279 ± 0.0810 |
| Bearing1_5 | 0.9502 | 0.8963 | 0.8981 ± 0.0563 |

每个seed相对距离基线的胜出数量分别为3/5、2/5和2/5；相对centered HMM分别为4/5、3/5和4/5。

从多seed均值看：

- GRU在Bearing1_2上略高于距离Level；
- Bearing1_1和1_3与距离Level基本相当；
- Bearing1_4和1_5低于距离Level；
- GRU高于HMM的优势主要来自训练bearings；
- Bearing1_5上GRU均值与HMM几乎相同，仅高0.0018。

因此，单seed结果中GRU描述性平均高于距离基线的现象不能视为稳定方法优势。

## 关键方法学发现

三个seed的next-step预测MSE非常接近，但隐藏状态距离产生的Level排序在Bearing1_4和1_5上差异较大。

这说明：

1. next-step prediction目标本身训练较稳定；
2. GRU hidden state没有被直接约束为健康轴；
3. 不同初始化可以学习相似预测函数，但形成不同的内部坐标表示；
4. 直接对隐藏状态使用欧氏距离，可能缺少表示可辨识性；
5. 继续增加Attention不会自动解决读出不稳定问题。

这是当前比“GRU是否胜过HMM”更重要的发现：问题开始从时序模型能力转向State Interpreter如何从时序表示中定义稳定、可解释的状态。

## 结论

第一版Predictive GRU在工程上可行，并稳定避免了HMM状态塌缩。它学习到的轨迹总体形状跨seed高度一致，但当前hidden-distance Level在未知bearing上的局部排序仍对初始化敏感，且没有稳定超过简单距离Level。

因此：

- HMM保留为离散时序基线；
- GRU保留为连续时序候选；
- 当前不进入Attention；
- 不应继续仅通过增加训练轮数或挑选最佳seed来报告结果；
- 下一步应先研究稳定的GRU状态读出方式。

## 下一步

保持三个已训练checkpoint不变，不重新训练模型，比较三种因果读出：

1. 当前hidden-state distance；
2. predicted-z相对早期参考的距离；
3. one-step prediction residual。

目标是判断初始化敏感性主要来自GRU动力学模型，还是来自对hidden state直接使用欧氏距离。如果output-space读出更稳定，再决定是否需要新的训练目标；如果仍不稳定，才考虑增加明确的时间排序或一致性约束。

## 统计与解释边界

- 三个seed不是三个独立设备样本；
- Bearing1_4用于checkpoint选择；
- Bearing1_5已经在前序实验中查看；
- 逐seed标准差描述初始化敏感性，不是总体泛化置信区间；
- normalized lifetime仅用于事后评价；
- ρ不代表物理损伤、RUL或RL决策收益；
- 不能根据训练bearing的稳定性推断未知bearing稳定性。

## Fallacy Scan（11/11）

| 检查项 | 状态 | 说明 |
|---|---|---|
| Simpson's paradox | 未发现 | 逐bearing和逐seed报告 |
| Ecological fallacy | 未发现 | 未由聚合均值推断单个bearing |
| Berkson's paradox | 注意 | 单一公开数据集和工况 |
| Collider bias | 不适用 | 未加入控制变量 |
| Base-rate neglect | 不适用 | 非分类实验 |
| Regression to the mean | 注意 | Bearing1_4因既有异常被重点研究 |
| Survivorship bias | 无法确认 | 数据集run筛选不由本实验验证 |
| Look-elsewhere effect | 注意 | 多方法、多bearing探索性比较 |
| Garden of forking paths | 注意 | 后续读出选择由当前已见数据引导 |
| Correlation ≠ causation | 注意 | 时间相关不能证明物理退化 |
| Reverse causality | 注意 | normalized lifetime不是独立损伤标签 |

## 软件与工件验证

- 三个checkpoint均存在；
- 训练历史900行；
- seed-bearing指标15行；
- step-level Level 1,848行；
- 轨迹pair比较15行；
- seed 20260916精确复现；
- 全量测试：79 passed；
- 无自动重试或事后调参。

## 工件

- 配置：`configs/xjtu_gru_predictive_multiseed_v1.json`
- 脚本：`scripts/run_gru_multiseed_stability.py`
- 协议：`docs/experiments/gru_predictive_multiseed_protocol_2026-09-16.md`
- 结果：`results/2026-09-16_gru_predictive_multiseed_stability/`

