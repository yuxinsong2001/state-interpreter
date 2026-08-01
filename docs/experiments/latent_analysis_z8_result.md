## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-01
- Verification Status: UNVERIFIED
- Version Label: exp_result_v1

# `z=8` Latent Space 首轮分析结果

## Experiment Result

- **ID**: `latent_analysis_z8_20260801`
- **Type**: analysis
- **Status**: completed
- **Working Directory**: `state-interpreter`
- **Checkpoint**: `runs/ae_baseline_z8_20260801/best_checkpoint.pt`
- **Output Directory**: `runs/latent_analysis_z8_20260801/`
- **Exit Code**: 0
- **执行说明**：本次代码、测试和分析由 Codex 在用户授权下执行，不表示用户已经亲自完成或理解了全部步骤。

## 研究问题

现有 AutoEncoder 产生的 `z∈R⁸` 是否包含随 bearing 退化过程变化、且能够迁移到未见 bearing 的结构？

## 防止信息泄漏的设置

- AutoEncoder 训练：`Bearing1_1–Bearing1_3`。
- validation：`Bearing1_4`。
- holdout test：`Bearing1_5`。
- 归一化统计量仅来自三个训练 bearings。
- PCA 仅在三个训练 bearings 的 latent 上拟合，再转换 validation 和 holdout。
- 健康中心仅使用三个训练 bearings 各自最早10%的样本，共46个样本。
- `Bearing1_4` 和 `Bearing1_5` 没有参与健康中心或 PCA 的拟合。

## 生成的工件

| 文件 | 作用 |
|---|---|
| `latent_trajectories.csv` | 616个样本的有序 `z`、时间、PCA、t-SNE、距离、`Δz` 和重建误差 |
| `bearing_summary.csv` | 每个 bearing 的汇总指标 |
| `analysis_report.json` | 机器可读的实验配置与结果 |
| `pca_trajectories.png` | 训练集拟合 PCA 后的轨迹 |
| `tsne_latent_space.png` | 仅用于定性观察的 t-SNE 图 |
| `health_center_distance.png` | 到训练早期健康中心的欧氏距离 |
| `delta_z.png` | 相邻测量之间的 latent 位移 |
| `reconstruction_mse.png` | 标准化 log-STFT 上的逐样本重建误差 |

## 核心结果

| Bearing | Split | 样本数 | 首点距离 | 末点距离 | 距离-时间 Spearman ρ | 平均重建 MSE |
|---|---:|---:|---:|---:|---:|---:|
| Bearing1_1 | train | 123 | 0.686 | 11.766 | 0.966 | 0.580 |
| Bearing1_2 | train | 161 | 0.764 | 11.813 | 0.982 | 1.059 |
| Bearing1_3 | train | 158 | 0.889 | 11.639 | 0.939 | 0.420 |
| Bearing1_4 | validation | 122 | 4.635 | 7.121 | **−0.799** | 0.616 |
| Bearing1_5 | holdout | 52 | 4.896 | 11.875 | 0.664 | 1.101 |

PCA 第一主成分解释约99.92%的训练 latent 方差，第二主成分约0.069%。这说明8维 latent 的主要变化几乎集中在一个方向上，但不能据此直接断言该方向就是纯健康变量；它也可能主要编码整体振动能量或其他随寿命共同变化的因素。

## 证据支持的判断

1. `z=8` 不是随机表示：训练 bearings 和 holdout `Bearing1_5` 中存在明显的时间有序变化。
2. 相邻位移 `Δz` 能识别突然状态变化，例如 `Bearing1_4` 最后一步的最大位移约为9.007。
3. 绝对健康中心距离不能直接作为通用 State Interpreter：两个未参与训练的 bearings 初始距离已经约为4.6–4.9，而训练 bearings 约为0.7–0.9。
4. `Bearing1_4` 的距离在大部分寿命中下降，导致负相关；这与“距离越大越退化”的统一解释相矛盾。
5. 重建误差在多数 bearings 的末期增加，但它仍是输入分布偏离指标，不等同于经过验证的健康状态。

## Interpretation Gate

**当前状态：不通过“直接采用全局健康中心距离”的门槛。**

当前结果允许继续设计 State Interpreter，但第一版不应仅输出：

```text
h_t = ||z_t - c_healthy||₂
```

更合理的下一候选是相对于每个 bearing 自身早期基线的变化：

```text
baseline_b = mean(z_b, early window)
relative_distance_t = ||z_t - baseline_b||₂
trend_t = slope(relative_distance over recent window)
movement_t = ||z_t - z_(t-1)||₂
```

这需要一个明确的 baseline/calibration 窗口，因此适用于同一设备随时间连续监测的场景；如果部署时只有互相独立的单次测量，则必须寻找跨 bearing 对齐方法或学习域不变表示。

## 限制

- XJTU-SY 没有在每一分钟提供真实健康阶段标签，本次只使用 normalized lifetime 作为时间代理，不能将其等同于真实损伤程度。
- 当前只分析一个工况 `35Hz12kN`。
- 健康窗口“最早10%”是实验假设，不是真实标注。
- t-SNE 只用于可视化，不能作为距离、聚类或阶段可分性的定量证据。
- 结果尚未做独立重复运行，因此 Material Passport 保持 `UNVERIFIED`。

## 下一步

实现并比较三个无需监督标签的候选 health indicator：

1. 全局健康中心距离（保留为失败/对照 baseline）；
2. 每个 bearing 自身早期窗口校准后的相对距离；
3. 相对距离 + 滑动趋势 + `Δz` 的轻量时序状态。

评价重点是 monotonicity、trendability、prognosability、平滑性，以及 validation/holdout 上的一致性；在这些简单方法明确失败前，不进入 GRU+Attention。

