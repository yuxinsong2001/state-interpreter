# Condition 3 source-source 条件 MMD 开发实验结果

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: execution + verification
- Origin Date: 2026-09-19
- Verification Status: COMPLETED / DEVELOPMENT ONLY / GATE FAILED
- Version Label: source_conditional_mmd_development_v1
- 预注册：[实验协议](xjtu_source_conditional_mmd_development_preregistration_2026-09-19.md)。
- 原始工件：[experiment_report.json](../../results/2026-09-19_source_conditional_mmd_development_v1/experiment_report.json)、[逐 seed 健康指标](../../results/2026-09-19_source_conditional_mmd_development_v1/per_seed_health_metrics.csv)、[identity probe](../../results/2026-09-19_source_conditional_mmd_development_v1/identity_probe.csv)。
- 配置 SHA-256：`1d9e9cfbf8b5f04bc2beaccdd5fd18fa8abf1a57b1efbef1c17cc921b826114b`；训练脚本 SHA-256：`41256a2500206ffa6016f2a0df8dc3c7f95df60a3c7f2df9dc7769593d55cc8d`。

## 问题和实验设计

检验仅在两个训练 bearing 之间按相同归一化时间区间对齐，能否比不对齐或全局 MMD 更好地保留完整生命周期的健康排序，并降低 bearing identity。三个分支同初始化、同采样索引、同训练轮数，三折 LOBO × 三随机种子；每折第三个 bearing 只用于开发评价。`Bearing3_4` 和 `Bearing3_5` 未读取。结果不是原版 CDAN/DSAN 的复现。

## 主要结果

下表为每个留出 bearing 上三个 seed 的健康预测与 normalized lifetime 的平均 Spearman ρ；括号内为正相关 seed 数（共 3）。

| 分支 | Bearing3_1 | Bearing3_2 | Bearing3_3 | identity 平均准确率 |
| --- | ---: | ---: | ---: | ---: |
| source-only | +0.089（1/3） | −0.580（0/3） | −0.450（0/3） | 0.907 |
| global MMD | +0.228（2/3） | −0.669（0/3） | −0.413（0/3） | 0.591 |
| conditional MMD | +0.308（3/3） | −0.748（0/3） | −0.407（0/3） | 0.605 |

Conditional MMD 的 identity 准确率 0.605 低于预设上限 0.80，因此 identity 子门槛通过；但健康子门槛的三项条件均失败：并非三个 bearing 均为正，零个 bearing 达到平均ρ≥0.5，B3_2/B3_3 均没有正相关 seed。联合门槛失败。相比配对 source-only，conditional MMD 在 B3_1 有所改善，B3_3 略有改善，B3_2 明显变差；不能称其解决了健康状态迁移。

Identity probe 的较低准确率仅表示在此探针设定下 bearing 身份较难区分，**不等于**健康信息被保留。normalized lifetime 也只是时间代理，非真实损伤标签；identity 是同一模型的开发 bearing 五时间块探针，并非额外独立外部集。

## 执行与工件核查

- 仅运行一次预注册命令，约 544 秒，正常退出；固定第 30 epoch checkpoint，没有按留出指标挑选 epoch。
- 输出完整：27 个 checkpoint、27 行逐 seed 指标、9 行 bearing×arm 汇总、810 行训练历史、9 行核尺度、162 行 identity probe、48,645 行轨迹。
- 运行后全量测试：`200 passed`。其中错误令牌测试改为核对已有输出快照不变，以兼容实验结果目录已存在的状态；未更改冻结训练脚本或配置。
- 报告注明未使用未来上下文，未以留出结果选 checkpoint；配置与七个冻结依赖哈希经程序核对。
- 此处没有根据结果修改 λ=0.1、五个时间区间、30 epochs、模型结构或推进门槛。既有 Source-DANN 工件未改写。

## 解释与下一步边界

本轮结果支持“对齐可压低 identity probe 准确率”，但**不支持**“该条件对齐足以产生跨 bearing 稳定健康排序”。B3_2 的明显退化说明时间区间并不自动等于相同物理健康阶段；也可能存在健康目标/工况差异或对齐过强，现有结果不能区分这些机制。下一步宜仅对已保存的开发轨迹和逐 seed 指标做只读失败诊断，并回到文献中寻找有物理/事件锚点的对齐方案。不能因这次失败而读取 B3_4/B3_5，不能事后挑选好 seed 或调参后把同一开发集当独立验证。

## AI 使用与限制

实验协议、实现及结果整理由 AI 协助；本报告保留失败结果和预注册门槛。三个开发 bearing 数量少，endpoint 不可视作独立设备重复；这里不声称统计显著性或真实剩余寿命预测能力。
