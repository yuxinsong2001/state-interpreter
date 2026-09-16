# 冻结GRU与持久性预测基线：比较结果

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：run / descriptive analysis
- Origin Date：2026-09-16
- Verification Status：既有30项预测MSE精确复现；因果前缀和基线恒等式核验通过
- Version Label：gru_persistence_comparison_result_v1

## 核心结果

当前两个GRU版本在全部30组主比较（2种训练目标×3个seed×5个bearing）中，单步MSE均高于持久性基线。每个bearing内对三个seed取平均，GRU误差是持久性误差的约1.52–4.53倍。加入排序目标基本没有改变这一结果。

因此，当前实验尚未证明GRU具有超过“保持当前embedding不变”的单步预测能力。此前较高的Level–时间相关性、平滑轨迹和持续下降的训练MSE，不能单独作为“学到了有用退化动力学”的证据。这不等于证明GRU普遍不可行，也不自动否定Level作为描述性指标的所有用途。

## 方法与评价口径

六个checkpoint来自已完成的时间排序实验，全程冻结，不拟合模型、不改变checkpoint选择。每个bearing仍使用前15点中心化；标准化使用checkpoint中保存的训练集统计量。两种预测在相同的标准化embedding空间比较：

- GRU：用截至t的历史产生下一时刻预测；
- persistence：直接令下一时刻预测等于当前观测，即`z_hat_(t+1)=z_t`。

主评分只使用目标step 25至末尾。程序保留从step 0开始的历史，使用`prediction[24:-1]`对齐`sequence[25:]`，不是从step 25才开始向GRU喂数据。末尾没有真实目标的预测被丢弃；误差不做平滑。normalized lifetime不参与误差计算。

定义`Skill = 1 - MSE_GRU/MSE_persistence`。正数表示GRU更好，0表示持平，负数表示GRU更差。三个seed均值表示各次模型指标的平均，不是预测集成效果。

## 主结果：目标step≥25

| Bearing | 目标数 | 持久性MSE | 原预测GRU MSE均值 | 原GRU/基线 | 排序GRU MSE均值 | 排序GRU/基线 |
|---|---:|---:|---:|---:|---:|---:|
| Bearing1_1 | 98 | 0.008338 | 0.025189 | 3.021 | 0.025045 | 3.004 |
| Bearing1_2 | 136 | 0.009941 | 0.019551 | 1.967 | 0.019637 | 1.975 |
| Bearing1_3 | 133 | 0.002706 | 0.011891 | 4.395 | 0.012050 | 4.454 |
| Bearing1_4 | 97 | 0.038775 | 0.058978 | 1.521 | 0.058870 | 1.518 |
| Bearing1_5 | 27 | 0.008694 | 0.039339 | 4.525 | 0.039329 | 4.524 |

Bearing1_1–1_3是训练数据；Bearing1_4参与checkpoint选择；Bearing1_5是反复分析过的旧留出数据，只有27个评分目标。表中任何一个结果都不应表述为新的严格盲测。

| Bearing | 原预测GRU Skill均值 ± seed标准差 | 排序GRU Skill均值 ± seed标准差 |
|---|---:|---:|
| Bearing1_4 | -0.5210 ± 0.0202 | -0.5182 ± 0.0193 |
| Bearing1_5 | -3.5247 ± 0.3019 | -3.5236 ± 0.2711 |

以原预测GRU为例，Bearing1_4的预测MSE比基线高约52%；Bearing1_5约为基线的4.52倍。不能将负Skill读作“健康程度为负”或“准确率为负”。

逐时刻胜率也偏低：两个GRU版本在Bearing1_4平均只有约7.22%的目标时刻优于基线，Bearing1_5约7.41%。所以当前劣势并非只来自某一个巨大的离群误差。所有30组MSE比较失败，也不意味着每一个时间点都失败。

## 历史口径澄清

上一轮README的“All metrics start at step 25”表述不够准确：Level指标从step 25评分，但`next_step_mse`实际上包含全序列step 1起的预测对。旧配置与旧结果保持原样；本报告明确更正这一解释。

本轮完整重算了30项历史全序列MSE，最大差异为0。全序列诊断下GRU也全部弱于持久性基线，说明主结论并非由step 25起点偶然造成。但全序列早期项使用了前15点校准参考，不应作为校准完成前的因果在线性能证据。

这里的精确复现使用专门的float32归约路径，与旧代码一致；新主表和全序列诊断表则统一用float64聚合float32平方误差，后者与历史值最多有约9.36e-9的末位差异。不能把诊断CSV本身误称为与旧CSV逐字节一致。

## 核验与产物

- 87项测试通过，包括手算序列的预测对齐、目标起点、恒定基线分母为0以及错误输入检查；
- 30项历史MSE复现最大绝对差异：0；
- 六模型×五bearing的前缀一致性最大差异：0；
- 持久性误差与相邻标准化embedding差分平方完全一致；
- 六个模型的标准化统计量完全相同，checkpoint、输入、训练配置和历史指标哈希核对通过；
- 原有受保护工件未修改，本轮未训练；
- 保存3,666行逐目标误差、60行逐seed指标（主区间与全序列两个口径）、20行汇总和30行核验记录；
- 生成`prediction_skill.png`，纵轴是误差比值的对数刻度，1为持久性基线，越低越好。

初次执行因自动审批服务“所选模型繁忙”未获放行，进程未启动。确认只读推理、全新输出目录及87项测试后，同一命令重试获准，约2.33秒完成实验计算。不存在失败训练或重复试验挑选问题。

## 当前决策与下一项建议

两臂均未通过“Bearing1_4和Bearing1_5上所有seed均有正Skill”的检查。当前不因时序网络更复杂就升级默认Level，也不把GRU残差直接称为已验证的故障指标。

训练bearing也弱于持久性，提示后续应先检查预测参数化或优化是否合理，而不仅归因于跨bearing泛化。下一步优先做冻结模型的增量诊断：比较`z_hat_(t+1)-z_t`与`z_(t+1)-z_t`，分维度检查系统偏移、变化方向和幅度，定位当前错误来自哪里。该诊断尚未执行。

如果证据支持模型连“保持当前值”都难以实现，再考虑残差预测候选`z_hat_(t+1)=z_t+delta_GRU`：让网络学习变化量，将修正头初始化为0，使初始模型精确等于持久性基线；之后仍用配对seed和相同评价口径检查正Skill。本轮尚未实现或训练此方案，不能预先保证它会改善。

即使残差预测将来改善，也仍需独立设备与物理事件评价，才能判断它是否适合作为State Interpreter或维护决策状态。

## 文件

- 配置：`configs/xjtu_gru_persistence_comparison_v1.json`
- 评价模块：`src/state_interpreter/forecast_evaluation.py`
- 脚本：`scripts/compare_gru_persistence.py`
- 完整结果：`results/2026-09-16_gru_persistence_comparison/`
- 主表：`summary_by_bearing.csv`（筛选`scope=primary`）
- 每个seed：`per_seed_bearing_metrics.csv`
- 原始误差对：`per_target_errors.csv`
- 核验：`integrity_checks.csv`、`experiment_report.json`
