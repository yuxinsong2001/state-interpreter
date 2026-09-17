# Residual GRU 终止性实验结果（2026-09-17）

## 结论摘要

Residual GRU通过了预先固定的最低推进门槛：在Bearing1_4和Bearing1_5上，三个seed的MSE Skill均为正。它在全部15个seed-bearing组合上也都优于persistence，并显著优于原direct GRU。因此“一步预测路线立即终止”的条件没有触发。

但这只支持继续验证Residual GRU，不支持把它直接升级为Health Level。Bearing1_4平均Skill只有2.20%，逐时刻胜率只有49.14%；B1_5虽有22.28%的平均Skill，但主评分只有27个目标且不是盲测。多数组合的中位逐步误差并未改善，MSE收益部分来自减少较大的少数误差。

## 实验固定条件

- 训练：Bearing1_1–Bearing1_3；
- checkpoint选择：Bearing1_4全序列next-step MSE；
- 旧holdout：Bearing1_5；
- seed：20260916、20260917、20260918；
- 与direct GRU相同的单层GRU、hidden dim 8、优化器和300轮上限；
- 主评价：target step ≥ 25；
- residual形式：`z_hat(t+1)=z(t)+delta_GRU(t)`；
- delta head初始为0，三个seed在训练前都逐值精确等于persistence。

## 结果

| Bearing | Direct MSE / persistence | Residual MSE / persistence | Residual Skill | 逐时刻胜率 |
|---|---:|---:|---:|---:|
| B1_1 | 3.021 | 0.948 | 0.052 | 0.367 |
| B1_2 | 1.967 | 0.935 | 0.065 | 0.387 |
| B1_3 | 4.395 | 0.861 | 0.139 | 0.541 |
| B1_4 | 1.521 | 0.978 | 0.022 | 0.491 |
| B1_5 | 4.525 | 0.777 | 0.223 | 0.605 |

表中均为三个seed的描述性均值。Residual三个seed的最佳epoch分别为45、75和12；训练分别在85、115和52轮结束。

### 预设门槛

| Bearing | 正Skill seed数 | 要求 | 结果 |
|---|---:|---:|---|
| B1_4 | 3/3 | 3/3 | 通过 |
| B1_5 | 3/3 | 3/3 | 通过 |

因此归档决策为`continue_prediction_route`，其含义是允许继续验证，而不是确认模型已可使用。

## 重要补充诊断

- 全部15个Residual组合的平均MSE Skill为正；
- 但只有6/15组合的中位逐时刻MSE低于persistence；
- B1_4三个seed的逐时刻胜率为48.45%–49.48%，说明2.2%的MSE提升主要来自对较大误差的控制，而不是多数时间点都更准；
- B1_5逐时刻胜率为59.26%–62.96%，但只有27个目标，且两个seed的中位误差仍高于persistence；
- 这说明Residual参数化有效抑制了direct GRU过大的运动，但典型时间点优势仍不稳定。

## 完整性与复核

- 全量测试：100 passed；
- direct GRU归档指标精确复现；
- 受保护输入、旧配置、旧结果和checkpoint未改变；
- 保存的三个Residual checkpoint重新加载后，对105个指标逐项复核，最大差异为0；
- 因果前缀最大差异为0；
- 三个Residual checkpoint、252行训练历史、30行主指标和2946行逐目标误差均已保存。

## 统计与推断边界

11类fallacy检查已覆盖。没有出现按bearing聚合后方向反转，但存在已查看样本和checkpoint选择造成的乐观偏差风险。未进行参数扫描，预设门槛仅是工程推进条件而非显著性检验。不能从三seed推断设备总体，也不能把预测相关性解释为健康因果语义。

## 下一步

不应立刻加入Attention。下一步应冻结Residual checkpoint，验证两件事：

1. 在未参与当前训练和选择的额外bearing或工况上，是否仍有正Skill；
2. 将Residual输出作为Movement/change signal，而不是直接作为Health Level，检查其与突变、退化阶段和现有`Level/Trend/Movement`的关系。

如果跨设备/跨工况优势消失，或稳健误差指标仍不优于persistence，应转向健康感知Encoder，而不是继续堆叠时序结构。
