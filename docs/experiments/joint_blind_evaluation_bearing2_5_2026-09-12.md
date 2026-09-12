# Bearing2_5 联合严格盲测结果（2026-09-12）

## 目的

在不重新调参的前提下，同时运行两条冻结流程：Arm A使用源工况归一化和源AutoEncoder直接泛化；Arm B使用目标工况归一化和目标工况AutoEncoder；两个分支使用相同的15步校准、10点窗口State Interpreter。

## 执行前安全门

- `Bearing2_5`此前没有生成signal statistics、latent、state或评价指标；
- 配置与两个checkpoint的SHA-256和冻结记录一致；
- 联合脚本已在`Bearing2_4`完成非盲演练；
- 修正pytest临时目录权限后，完整测试为`66 passed`；
- 两个分支共享同一次原始数据/STFT物化。

## 正式结果

| 指标 | Arm A | Arm B | Arm B − Arm A |
|---|---:|---:|---:|
| Level–normalized lifetime Spearman ρ | 0.946183 | 0.964242 | +0.018059 |
| READY状态数 | 324 | 324 | — |
| READY比例 | 0.955752 | 0.955752 | — |
| Level first | 0.409113 | 0.206105 | — |
| Level last | 10.645162 | 15.846568 | — |
| Level smoothness | 0.711685 | 0.738959 | — |
| Trend noise | 0.009542 | 0.013022 | — |
| Movement max | 1.944285（step 121） | 1.380093（step 281） | — |

原始测量数为339；前15个测量用于每个分支的校准，因此每个分支产生324个正式状态。

## 结论

按照预注册阈值`ρ ≥ 0.7`，Arm A和Arm B都属于强支持：两条冻结流程都能在此前未知的Bearing2_5上产生与生命周期高度同序的Level。Arm B仅比Arm A高`0.0181`，因此结果支持“Encoder适配带来小幅额外收益”，不支持“Arm B显著优于Arm A”的强结论。

曲线显示Arm A在约step 200后接近平台，而Arm B在后半段继续上升；Arm B的Trend和Movement后段更活跃。该差异可作为未来研究问题，但不能使用Bearing2_5重新选择参数或checkpoint。

## 解释边界

- 只有一个严格盲测bearing，不能估计跨bearing分布或统计显著性；
- Spearman ρ衡量时间排序，不证明Level等于真实物理损伤；
- 两个分支使用不同归一化统计量，重建MSE不作为主要优劣证据；
- Bearing2_5从此不得再用于模型选择或调参；
- 不允许通过重复运行或修改Interpreter参数优化本次结果。

## 工件

- `runs/joint_blind_bearing2_5_20260912/joint_states.csv`
- `runs/joint_blind_bearing2_5_20260912/joint_state_plot.png`
- `runs/joint_blind_bearing2_5_20260912/joint_report.json`
- `records/xjtu_cross_condition_v2_1/2026-09-12_joint_blind_evaluation.json`

原始`joint_report.json`中的日期和experiment ID沿用了脚本内旧的`2026-08-02`字符串。为保留原始实验工件，不做事后重写；实际日期由结果目录、文件时间和独立record记录。
