# Arm B 目标工况 Encoder：开发集 State Interpreter 结果

## 本步回答的问题

在 `37.5Hz11kN` 上重新训练同结构 `z=8` AutoEncoder 后，保持 State Interpreter 完全不变，能否从新的 latent space 得到随寿命推进的状态轨迹？本步不是最终盲测，只使用 `Bearing2_1`–`Bearing2_4`；`Bearing2_5` 未读取。

## 固定设置

- Encoder：Arm B epoch 4 checkpoint
- State Interpreter：`RelativeTemporalStateInterpreter`
- latent 维数8，校准长度15，时间窗口10
- 输出：`level, trend, movement`
- 主要指标：`level` 与 normalized lifetime 的 Spearman ρ

## 开发集结果

| Bearing | Arm A ρ | Arm B ρ | B − A |
|---|---:|---:|---:|
| Bearing2_1 | 0.887 | 0.914 | +0.026 |
| Bearing2_2 | 0.991 | 0.994 | +0.003 |
| Bearing2_3 | 0.974 | 0.979 | +0.005 |
| Bearing2_4 | 0.681 | 0.993 | +0.312 |

四个 bearing 的平均 `Δρ` 为 `+0.087`。目标工况训练后的 Encoder 在全部四个开发 bearing 上都没有降低主要指标，其中对验证 bearing `Bearing2_4` 的改善最明显。

## 如何理解

结果阶段性支持：固定的相对时间 State Interpreter 可以处理目标工况上重新学习的 latent space；Encoder 的目标工况适配尤其改善了原先较弱的 `Bearing2_4`。

但目前不能说 Arm B 已经在未知设备上优于 Arm A，因为 `Bearing2_1`–`Bearing2_3` 参与过 Encoder 训练，`Bearing2_4` 参与过 checkpoint 选择。真正决定泛化结论的是尚未读取的 `Bearing2_5` 联合盲测。两分支使用不同归一化统计量，重建 MSE 的绝对值不能作为严格优劣依据。

## 完整性检查

- 共处理1,227个样本，全量测试62项通过。
- bearing 字段和 source path 中 `Bearing2_5` 均为0行。
- v2.1 配置哈希保持为 `788b4c25ae32c4eee4f4237df372f0087f57f30608a00a3cb09e2acb217ceae3`。
- Arm B checkpoint 哈希保持为 `479f66d07916fc43556da3b74a4eae35c42f58424ff1307b76a8dafcca4f818c`。

## 下一步

实现并在伪造/开发数据上测试联合盲测脚本。脚本应一次读取 `Bearing2_5`，同时计算 Arm A 和 Arm B；当前阶段仍不读取盲测数据。
