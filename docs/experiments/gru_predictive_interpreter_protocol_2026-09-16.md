# 第一版Predictive GRU State Interpreter实验协议

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：plan
- Origin Date：2026-09-16
- Verification Status：UNVERIFIED
- Version Label：gru_predictive_protocol_v1

## Experiment Overview

- **Title**：基于下一时刻embedding预测的因果GRU State Interpreter
- **Objective**：检查小型GRU学习时间动态后产生的连续隐藏状态，是否比centered HMM更稳定，同时保持跨bearing时间排序。
- **Hypothesis**：GRU可以避免HMM离散状态塌缩，但是否优于距离Level由实验决定。
- **Type**：training + analysis

## 数据与划分

| 角色 | Bearing | 权限 |
|---|---|---|
| Training | Bearing1_1–Bearing1_3 | 拟合中心化后的特征标准化与GRU参数 |
| Validation | Bearing1_4 | 仅选择最低next-step MSE checkpoint |
| Old holdout | Bearing1_5 | 仅在checkpoint锁定后评价 |

所有bearing均已在过去实验中查看，因此本次是探索性实验，不是盲测。

## 输入与训练目标

每个bearing先执行：

```text
z_centered(t) = z(t) - mean(z(1:15))
```

随后仅使用训练bearing拟合逐维均值和标准差。GRU在时间`t`读取当前及过去embedding，并预测`t+1`的标准化centered embedding。损失为全部训练序列的加权next-step MSE。normalized lifetime不参与训练。

## 模型

```text
centered standardized z_t (8D)
→ one-layer causal GRU (hidden=8)
→ hidden state h_t
→ linear prediction head
→ predicted z_(t+1) (8D)
```

模型规模故意保持很小，不加入Attention、dropout或多层结构。

## 连续状态读出

GRU训练后冻结。对每个bearing计算前15个隐藏状态均值`h_ref`，定义：

```text
raw_level(t) = ||h_t - h_ref||₂
level(t) = 最近最多10个raw_level的均值
```

评分从step 15开始。该Level是GRU内部动态状态相对于早期参考的距离，不是物理损伤百分比。

## 比较与主要指标

- 距离Level；
- bearing-centered HMM expected stage；
- Predictive GRU Level。

主要描述性指标为逐bearing Level–normalized lifetime Spearman ρ。辅助检查包括first-last delta、step标准差、backward-step比例、next-step MSE和是否输出常数。

## 固定设置

- seed：20260916；
- calibration steps：15；
- temporal window：10；
- hidden dim：8；
- optimizer：Adam；
- learning rate：1e-3；
- weight decay：1e-5；
- max epochs：300；
- patience：40；
- gradient clipping：1.0。

本轮不根据Bearing1_5结果重新调参。如果单种子结果可运行且非塌缩，下一阶段再决定是否开展多随机种子实验。

## 解释边界

- next-step预测成功不自动代表健康语义；
- Spearman相关只评价时间排序；
- GRU可能学习时间相关结构，但不能据此宣称学习了物理损伤；
- 单随机种子结果不能用于稳定性结论；
- 与HMM和距离基线的比较属于已见数据上的探索性方法筛选。
