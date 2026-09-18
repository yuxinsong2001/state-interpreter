# Condition 3 冻结式开发诊断结果（2026-09-18）

## 目的与边界

本实验使用预注册且完全冻结的 Relative State Interpreter v2，首次读取 XJTU-SY Condition 3 的开发 bearing：Bearing3_1–Bearing3_3。Bearing3_4（验证）和 Bearing3_5（一次性测试）均未读取。

冻结内容包括 Condition 1 AutoEncoder 与归一化、训练 bearing 拟合的 signed degradation axis、三个 Residual-GRU checkpoint、`calibration_steps=15`、`temporal_window=10`，以及由训练数据固定的 Movement 阈值 15.0227。

本阶段回答的问题是：现有系统在不重训、不重新标定的情况下，能否直接泛化到第三种运行工况。

## 结果

| Bearing | 样本数 | Level–lifetime Spearman ρ | 首末 Level 差 | Movement 超阈值比例 | Movement 中位数 | 重建 MSE 均值 |
|---|---:|---:|---:|---:|---:|---:|
| Bearing3_1 | 2538 | -0.391 | +12.731 | 0.080% | 1.206 | 0.330 |
| Bearing3_2 | 2496 | +0.894 | +9.461 | 13.962% | 3.362 | 0.632 |
| Bearing3_3 | 371 | -0.564 | +7.463 | 0.867% | 1.073 | 0.622 |

三个 bearing 的状态均为有限值，工程流程完整运行。所有轨迹的末端 Level 都高于起点，但 Bearing3_1 和 Bearing3_3 的全生命周期排序为负：它们在大部分寿命中先沿负方向缓慢移动，临近末端才快速转为正方向。因此，仅看首末差会掩盖主要生命周期中的反向轨迹。

Bearing3_2 的 Level 排序较强，但 Movement 超阈值比例达到 13.962%，而阈值来自 Condition 1 训练数据。其 Movement 在较长区间内整体偏高，不是少数孤立末端峰值。这说明旧 Movement 标尺在该 bearing 上发生明显分布偏移，不能直接把这些超阈值点解释为故障事件。

## 结论

冻结方案的“直接跨工况泛化”在 Condition 3 开发集上不成立：只有 1/3 bearing 保持强正向 Level 排序，且 Movement 的跨工况标定在 Bearing3_2 上失效。

这不是“模型对 Condition 3 完全无效”。三个 bearing 均出现正的首末 Level 差，末端变化可见；问题是共同退化轴的方向与速度不能在整个生命周期内稳定迁移，且不同 bearing 的 Movement 尺度不一致。证据更支持表示/域偏移与跨工况标定问题，而不是简单调一个时间窗口即可解决的问题。

因此暂不读取 Bearing3_4 和 Bearing3_5，也不宣布最终失败。下一步应仅使用 Bearing3_1–Bearing3_3 比较受控的适配候选，例如重新拟合 Condition 3 的 Encoder/退化轴，或建立训练工况条件化的校准；确定唯一方案并冻结后，再由 Bearing3_4 验证。Bearing3_5继续保留为一次性最终测试。

## 有效性与误读检查

- 不汇总三个 bearing 为单一相关系数，避免异质轨迹造成 Simpson 式误读。
- bearing 是评价单位；时间点不是独立设备样本，不据此夸大样本量。
- 数据是实验室 run-to-failure 轨迹，不能直接外推到存在删失和维护干预的现场数据。
- 指标、阈值、模型和数据划分在读取 Condition 3 前已经锁定，没有依据本次结果事后挑选。
- 相关性只描述时间排序，不证明物理损伤因果，也不等于健康百分比、故障阶段或 RUL。
- Movement 的基准率随 bearing 显著变化，超阈值只能称为模型意外事件，不能直接称为故障。

## 可复现性

- 结果目录：`results/2026-09-18_condition3_frozen_development/`
- 完成记录：`records/xjtu_condition3_frozen_v2/development_completed.json`
- 运行耗时：1013.95 秒
- 回归测试：126 passed

