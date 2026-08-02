## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-02
- Verification Status: UNVERIFIED
- Version Label: exp_result_v1

# 分支A：旧系统直接泛化开发实验

## Experiment Result

- **ID**：`arm_a_direct_generalization_dev_20260802`
- **类型**：analysis
- **状态**：completed
- **目标工况**：`37.5Hz11kN`
- **开发bearing**：`Bearing2_1–Bearing2_4`
- **严格盲测**：`Bearing2_5`，未读取
- **运行时间**：74.4秒（分析进程内部计时）
- **退出码**：0

本步骤由 Codex 在用户授权下实现、测试和运行，不表示用户已经亲自完成或完全理解全部代码。

## 固定输入

- 使用 `35Hz12kN` 训练的 v1 AutoEncoder；
- 使用checkpoint内保存的v1归一化均值和标准差；
- AutoEncoder没有训练或更新；
- 没有在目标工况重新拟合归一化；
- State Interpreter固定为15步校准、10点窗口和 `[level, trend, movement]`；
- v2.1配置SHA-256：`788b4c25ae32c4eee4f4237df372f0087f57f30608a00a3cb09e2acb217ceae3`；
- v1 checkpoint SHA-256：`6edb626d1961cb1b1a7a2c32c52c9c96b0100cabb712cda4d79ab3c2a2e7695a`。

## 结果

| Bearing | 总样本 | READY | Level Spearman ρ | Smoothness | 首个level | 最后level | 最大movement |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bearing2_1 | 491 | 476 | 0.887 | 0.690 | 0.373 | 15.833 | 5.637 |
| Bearing2_2 | 161 | 146 | 0.991 | 0.713 | 0.058 | 11.482 | 2.280 |
| Bearing2_3 | 533 | 518 | 0.974 | 0.689 | 0.323 | 15.247 | 2.720 |
| Bearing2_4 | 42 | 27 | 0.681 | 0.816 | 0.319 | 13.471 | 7.503 |

前三个开发/训练bearing平均ρ为0.951；四个开发bearing平均ρ为0.883。

## 曲线解释

- 四个bearing的level最终都明显高于校准后初值，说明旧系统在新工况中仍能表示早期到后期的总体变化；
- `Bearing2_1` 在约step 450后快速上升；
- `Bearing2_2` 在约step 55–100持续上升后进入平台；
- `Bearing2_3` 存在多段变化，在约step 330–365快速上升后进入平台；
- `Bearing2_4` 寿命最短，约step 30发生大幅movement和level快速上升；其READY点只有27个，ρ=0.681低于预注册的0.70“较强支持”阈值。

不同bearing的轨迹形状差异明显，说明单一单调曲线不足以描述所有退化形式，而level、trend和movement分别保留了累计偏离、局部发展方向和突变信息。

## 可以得出的结论

在开发集上，旧AutoEncoder、旧归一化和固定State Interpreter无需目标工况训练，也能在四个bearing上产生正向level趋势。这是完整系统具有一定直接跨工况能力的初步证据。

## 不能得出的结论

- 不能宣称分支A已经通过最终跨工况测试；
- 不能把 normalized lifetime 当作真实damage标签；
- 不能依据开发结果修改15/10参数或成功阈值；
- 不能与尚未完成的分支B比较；
- 不能读取 `Bearing2_5`，直到两分支和联合评价代码全部冻结。

## 输出工件

- `development_states.csv`：1,227行逐时刻latent、重建误差和状态；
- `bearing_summary.csv`：四个bearing的指标；
- `development_report.json`：输入哈希、范围声明和结果；
- `development_state_plot.png`：level、trend和movement曲线。

## 运行异常

首次启动在创建输出目录前因沙箱写权限退出，没有读取任何bearing，也没有留下结果。获得写权限后以完全相同的命令重新运行并成功完成；模型和实验参数没有变化。

## 下一步

实现分支B的独立训练入口，在 `Bearing2_1–2_3` 上拟合目标工况归一化并训练同结构AutoEncoder，使用 `Bearing2_4` 选择checkpoint。仍不得读取 `Bearing2_5`。

