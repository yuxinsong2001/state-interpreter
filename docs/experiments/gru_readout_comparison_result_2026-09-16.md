# 固定 GRU checkpoint 的状态读出对照结果

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：execute
- Origin Date：2026-09-16
- Verification Status：VERIFIED
- Version Label：gru_readout_comparison_result_v1

## 研究目的

上一轮多随机种子实验发现：三个 Predictive GRU 的 next-step prediction MSE 很接近，但由 hidden state 欧氏距离得到的 Level 在 Bearing1_4 和 Bearing1_5 上仍有明显差异。本实验冻结三个既有 checkpoint，只改变 State Interpreter 的读出方式，以判断问题是否主要来自 hidden-state distance 的定义。

## 固定条件

- checkpoint：seed 20260916、20260917、20260918；
- 不重新训练、不选择最佳 seed；
- 使用各 checkpoint 内保存的标准化统计量；
- 每个 bearing 使用前 15 点中心化；
- 使用 10 点移动平均；
- 三种读出统一从 step 25 开始评分；
- normalized lifetime 只用于描述性评价，不参与训练或读出。

## 对照的三种读出

1. `hidden_distance`：当前 hidden state 到早期 hidden reference 的欧氏距离。
2. `predicted_z_distance`：在上一步产生的因果预测 `z_hat_t` 到早期预测参考的欧氏距离。
3. `prediction_residual`：实际 `z_t` 与上一步预测 `z_hat_t` 之间的误差范数。

## 结果核验

- 实验状态：`completed`；
- 验证类型：`fixed_checkpoint_ablation`；
- hidden-distance 逐点复现最大绝对误差：`0.0`；
- per-step 行数：1,848；
- seed-bearing-readout 指标行数：45；
- 所有方法、所有 seed 和 bearing 的 collapse 数均为 0；
- 三个 checkpoint、输入文件和配置文件的 SHA-256 已写入 `experiment_report.json`。

## 主要结果

### 各 bearing 的 Spearman ρ 均值与 seed 标准差

| Bearing | Hidden distance | Predicted-z distance | Prediction residual |
|---|---:|---:|---:|
| Bearing1_1 | 0.952 ± 0.001 | 0.952 ± 0.001 | 0.826 ± 0.012 |
| Bearing1_2 | 1.000 ± 0.000 | 0.999 ± 0.000 | -0.423 ± 0.019 |
| Bearing1_3 | 0.963 ± 0.002 | 0.960 ± 0.002 | 0.248 ± 0.119 |
| Bearing1_4 | 0.828 ± 0.063 | 0.807 ± 0.070 | -0.438 ± 0.028 |
| Bearing1_5 | 0.949 ± 0.032 | 0.925 ± 0.024 | 0.670 ± 0.010 |

这些 ρ 使用统一的 step 25 评分起点，因此不应直接与此前从 step 15 开始计算的数字混合比较。

### 聚合比较

| Readout | 五个 bearing 的描述性平均 ρ | B1_4/B1_5 的平均 seed 标准差 | 全部 seed-bearing 中最低 ρ |
|---|---:|---:|---:|
| Hidden distance | 0.938 | 0.04759 | 0.756 |
| Predicted-z distance | 0.929 | 0.04743 | 0.730 |
| Prediction residual | 0.177 | 0.01897 | -0.470 |

## 结论

### 1. Predicted-z distance 没有解决初始化敏感性

它在 Bearing1_4 和 Bearing1_5 上的平均 seed 标准差为 0.04743，与 hidden distance 的 0.04759 几乎相同；同时其描述性平均 ρ 和最低 ρ 略低。因此当前证据不支持用 predicted-z distance 替代 hidden-distance Level。

### 2. 问题不只是 hidden coordinate 的选择

如果不稳定只来自 hidden state 坐标不可辨识，转到 prediction output space 后应明显改善。但实验没有出现这种改善。这表明不同初始化还可能影响局部预测轨迹的时间排序，或者 next-step prediction 目标本身没有约束模型学习稳定的健康方向。

### 3. Prediction residual 不适合作为单调 Health Level

Residual 在不同 bearing 上既可能正相关，也可能负相关，例如 Bearing1_4 为 -0.438、Bearing1_5 为 0.670。它反映“当前观测有多出乎模型预期”，而不是“设备距离健康状态有多远”。因此不能把它作为 Level 的直接替代，但可以作为 Movement、change score 或 anomaly score 的候选补充量。

## 当前架构决策

- 暂不引入 Attention，因为更复杂的序列模块不会自动产生稳定且有健康语义的读出。
- 当前最稳妥的核心 Level 仍是 bearing-relative latent distance。
- Predictive GRU residual 可作为辅助变化信号继续验证，但不承担 Health Level 的语义。
- 若继续研究 learned Level，应在训练目标中显式加入与状态语义一致的约束，例如 temporal consistency、ordinal/ranking 或弱单调约束，而不是只优化 next-step MSE。

## 限制

- 本实验使用的 bearing 和 checkpoint 均已在此前分析中查看，不是新的盲测；
- 三个 seed 是同一批设备数据上的重复模型，不是三个独立设备样本；
- Spearman ρ 只反映时间排序，不能证明物理损伤语义；
- residual 是否适合作为异常指标仍需事件标签或故障时刻证据验证。

## 产物

- `configs/xjtu_gru_readout_comparison_v1.json`
- `scripts/compare_gru_state_readouts.py`
- `docs/experiments/gru_readout_comparison_protocol_2026-09-16.md`
- `results/2026-09-16_gru_readout_comparison/experiment_report.json`
- `results/2026-09-16_gru_readout_comparison/per_step_readouts.csv`
- `results/2026-09-16_gru_readout_comparison/per_seed_bearing_metrics.csv`
- `results/2026-09-16_gru_readout_comparison/summary_by_readout_bearing.csv`
- `results/2026-09-16_gru_readout_comparison/trajectory_consistency.csv`
- `results/2026-09-16_gru_readout_comparison/readout_aggregate.csv`

