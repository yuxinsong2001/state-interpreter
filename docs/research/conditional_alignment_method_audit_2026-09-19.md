# 条件对齐方法审计：从 CDAN/DSAN 到当前 LOBO 问题

## 研究边界

当前任务是用两个训练 bearing 学习可迁移的连续健康表示，对整条留出 bearing 做健康排序；`Bearing3_4`/`Bearing3_5`仍受保护。此前 Source-DANN 在开发 bearing 上降低了一部分 identity，但未通过健康及 identity 联合门槛。五等分生命周期诊断只有 6/15 个留出折×区间同时出现健康排序改善与 identity 下降；这仅提示全局对齐的效果不均匀，**没有证明条件对齐一定有效**。详见[阶段诊断](../experiments/xjtu_source_dann_lifecycle_diagnosis_2026-09-19.md)。

## 核对的成熟方法

| 方法 | 原文机制 | 与当前问题的关键差异 | 决策 |
|---|---|---|---|
| CDAN | 判别器联合使用特征与分类预测，缓解多模态分布的错误匹配；原问题在训练时拥有带标签 source 和无标签 target。 | 当前没有可在训练时读取的留出 bearing；健康目标是连续 normalized lifetime，不是分类概率。 | 不直接移植原模型或把连续时间强行伪装成故障类别。 |
| DSAN/LMMD | 利用类别信息和 target 伪标签，只对齐对应类别的 source/target 子域。 | 无 target 数据可供伪标记；五个时间区间并非已验证的物理退化阶段。 | 借鉴“对应区间而非全局混合”的思想，但不能称为原版 DSAN 复现。 |
| Source-source 条件 MMD | 在两个**训练** bearing 的相同时间进度区间内比较 embedding 分布。 | 是为严格 LOBO 改写的探索性 source-domain regularizer，并非上述论文已验证的 bearing RUL 方法。 | 可以设计受控开发实验，必须同时有无对齐及全局 MMD 对照。 |

CDAN 的原论文与[公开实现](https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/alignment/cdan.py)均围绕分类预测条件化判别器；DSAN 的原论文明确使用 local MMD 对齐同类子域，并以未标记目标域为前提。参见[CDAN 原论文](https://papers.nips.cc/paper/2018/file/ab88b15733f543179858600245108dd8-Paper.pdf)、[DSAN 原论文](https://jd92.wang/assets/files/a24tnnls20.pdf)及[作者代码仓库](https://github.com/easezyc/deep-transfer-learning)。这里的 source-source 条件 MMD 是我们的**方法迁移假设**，不是文献原样算法，也不借用原论文性能结论。

## 标签语义与数据泄漏

- `normalized lifetime` 只能称为时间进度代理：长寿命与短寿命 bearing 的同一百分位不必代表相同物理损伤。区间名称必须写成“时间进度区间”，不能写成健康/损伤阶段。
- 每折的区间边界、MMD 带宽、尺度及训练采样只可由两个训练 bearing 决定；留出的第三个 bearing 不参与训练、伪标签、带宽拟合、checkpoint 选择或损失权重调整。
- B3_4/B3_5 继续不读取；此前对 B3_1–B3_3 的探索意味着后续仍是**开发实验**，不能追认成独立确认性检验。
- 身份信息下降本身不代表健康表示改善；必须和完整 bearing 的健康排序、seed 一致性共同判断。

## 方法决策

**有条件地继续**一个三分支、三折 LOBO 的 source-source MMD 对照：source-only、global MMD、time-bin conditional MMD。这样可以检验“对齐本身”与“按时间区间对齐”的增量，而不是只把失败的 DANN 换个名字。实验方案见[条件 MMD 开发协议草案](../experiments/xjtu_source_conditional_mmd_protocol_2026-09-19.md)。

本步只完成文献核对与研究设计；**未实现损失、未训练、未得到新实验结果**。协议仍是草案，下一步须用训练数据的 no-held-out preflight 检查每个区间的配对样本数、核尺度与损失数值，再冻结配置和哈希。失败时保留负结果，不因局部区间改善开启保护集。

## 局限与 AI 使用说明

本审计由 AI 辅助阅读论文与公开代码并形成与项目约束相匹配的推论；“source-source time-bin MMD”是本项目的候选设计，不是论文已有结论。未重新复现 CDAN/DSAN，也未检验物理阶段标签的真实性。引文已核对原论文与公开仓库，实验设计仍须用本项目数据独立验证。
