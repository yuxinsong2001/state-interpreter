# 冻结 GRU 潜空间增量诊断结果（2026-09-17）

## 结论摘要

当前 GRU 低于 persistence baseline 的主要原因不是一个可用常数修正的固定偏差，而是预测增量的方向不稳定、幅度随 bearing 明显失配。事后 oracle 去偏后，30/30 组比较仍未超过 persistence。control 与 ranking 的诊断结果几乎重合，说明上一轮时间排序损失没有修复这一动态预测问题。

## 实验范围

- 冻结 checkpoint：control/ranking × seed 20260916–20260918；
- bearing：Bearing1_1–Bearing1_5；
- 主评分：target step ≥ 25；
- 不训练、不调参、不重新选择 checkpoint；
- 完整复现上一轮30项GRU MSE，最大差异为0；
- 因果前缀最大差异和输入/config/checkpoint/旧结果哈希均通过检查；
- 全量测试：94 passed。

## 聚合结果

| Arm | Bearing | Raw MSE / persistence | Oracle去偏 MSE / persistence | Bias占比 | 预测/真实增量RMS | Flattened cosine | 正内积时刻比例 |
|---|---|---:|---:|---:|---:|---:|---:|
| control | B1_1 | 3.021 | 2.868 | 0.051 | 1.538 | 0.112 | 0.575 |
| control | B1_2 | 1.967 | 1.508 | 0.233 | 1.139 | 0.145 | 0.441 |
| control | B1_3 | 4.395 | 4.043 | 0.078 | 2.107 | 0.249 | 0.474 |
| control | B1_4 | 1.521 | 1.484 | 0.024 | 0.673 | -0.051 | 0.584 |
| control | B1_5 | 4.525 | 4.103 | 0.093 | 1.760 | -0.121 | 0.642 |
| ranking | B1_1 | 3.004 | 2.859 | 0.048 | 1.531 | 0.111 | 0.571 |
| ranking | B1_2 | 1.975 | 1.504 | 0.239 | 1.140 | 0.142 | 0.436 |
| ranking | B1_3 | 4.454 | 4.122 | 0.073 | 2.118 | 0.245 | 0.474 |
| ranking | B1_4 | 1.518 | 1.482 | 0.024 | 0.670 | -0.052 | 0.584 |
| ranking | B1_5 | 4.524 | 4.102 | 0.093 | 1.760 | -0.121 | 0.642 |

表中为三个seed的描述性均值，不是模型集成，也不是设备总体推断。

## 结果解释

### 1. 固定偏差不是主因

- 30组中没有一组达到预先定义的“bias占raw MSE至少50%”；
- 全部30组bias占比平均为9.58%；
- oracle去偏后的MSE比值平均从3.090降到2.807，但仍有0/30低于1；
- 因此不能用简单常数bias calibration解释或修复失败。

### 2. 幅度失配具有明显bearing依赖

- 24/30组触发幅度失配标记；
- B1_3预测增量约为真实RMS的2.11倍，B1_5约1.76倍；
- B1_4反而只有约0.67倍；
- 只有B1_2落在预设的0.8–1.25描述区间。

这不是统一乘一个全局系数就能稳定解决的问题。

### 3. 方向信息弱且不稳定

- 24/30组触发方向较差标记；
- B1_4与B1_5的整体flattened cosine为负；
- B1_2与B1_3虽然整体cosine为正，但单步正内积比例低于50%；
- 全部8个latent维度的跨组合平均Pearson约为-0.03至-0.09，说明逐时刻增量相关性整体接近零或略负。

因此，仅把预测结果减去平均偏差不能恢复可靠动态方向。

### 4. 排序目标没有改变误差结构

control和ranking在三类图上几乎重合。时间排序弱监督影响了hidden-distance Level，但没有带来更可靠的下一步latent增量预测。

## 对 residual GRU 的判断

本实验提供了“值得做严格对照实验”的依据，但没有证明 residual GRU 一定有效：

- 支持理由：当前模型常产生过大的非零运动；显式形式 `z_hat(t+1)=z(t)+delta_GRU(t)` 可以把零增量/persistence作为结构基线，并让模型只学习额外变化；
- 限制理由：4/5个bearing触发方向问题，残差参数化本身不能保证方向正确。

所以下一步应把 residual GRU 当作新的受控候选，而不是直接替换默认 State Interpreter。保持数据划分、标准化、网络容量、训练轮数、三个seed和评分范围不变，与当前 direct GRU 及 persistence 比较。若仍不能获得正Skill，则停止把单步预测作为Health Level的主要训练依据。

## 文件

- `per_seed_bearing_metrics.csv`：30组主诊断；
- `per_dimension_metrics.csv`：240组分维度诊断；
- `per_target_metrics.csv`：2946行逐目标结果；
- `summary_by_bearing.csv`：10行三seed聚合；
- `integrity_checks.csv`：30项复现与因果检查；
- `increment_diagnosis.png`：误差、幅度和方向图；
- `experiment_report.json`：配置、哈希、环境、限制与完整汇总。

## 限制

这些bearing均已查看过，B1_4参与checkpoint选择，B1_5也已反复分析；三seed不能替代多设备样本。oracle去偏使用整段真实目标，不可部署。本实验只分析latent单步预测，不证明健康语义或维护决策价值。
