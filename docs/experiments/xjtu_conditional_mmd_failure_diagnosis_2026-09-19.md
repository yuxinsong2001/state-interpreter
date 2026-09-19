# Condition 3 条件 MMD 失败轨迹诊断

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate / exploratory read-only diagnosis
- Origin Date: 2026-09-19
- Verification Status: ANALYZED, not independently reproduced
- Version Label: conditional_mmd_failure_diagnosis_v1
- 来源：[冻结开发实验结果](xjtu_source_conditional_mmd_development_result_2026-09-19.md)。
- 诊断输入仅为该实验已保存的 `experiment_report.json`、`health_trajectories.csv`、`per_seed_health_metrics.csv`；SHA-256 保存在[诊断审计文件](../../results/2026-09-19_conditional_mmd_failure_diagnosis_v1/diagnosis_report.json)。没有加载原始数据、checkpoint或保护bearing。
- 图：[三分支三bearing逐seed原始曲线](../../results/2026-09-19_conditional_mmd_failure_diagnosis_v1/development_trajectories.png)。原始预测仅为绘图下采样，未平滑；统计使用所有点。

## 问题与方法

问题是负相关究竟属于全程方向翻转，还是部分时间段的阶段错位。将已保存轨迹按归一化测量时间均分为五区间，逐 arm×seed×bearing 计算区间内 Spearman、区间预测均值，以及首末区间均值变化。脚本逐条重算全生命周期 Spearman 并与实验CSV核对；27条全部一致。五区间只是诊断刻度，不是真实损伤阶段，且这是事后探索性分析，不能重开预注册推进门槛。

## 观察结果

| 留出bearing | Conditional MMD 整段平均ρ | 逐seed整段方向 | 区间0→4预测均值变化（3 seed平均） | 主要轨迹形态 |
| --- | ---: | --- | ---: | --- |
| B3_1 | +0.308 | 3/3正 | +0.087 | 中段波动，末端上升；整体弱正相关 |
| B3_2 | −0.748 | 0/3正 | −0.351 | 前段上升，中段到后段持续下降，方向与训练目标相反 |
| B3_3 | −0.407 | 0/3正 | +0.063 | 前中段下降，临近末端急升；非整体反向 |

B3_2 conditional MMD 的区间内平均ρ从第0至第4区间分别为`+0.260、−0.080、−0.580、−0.530、+0.280`；区间预测均值为`0.420、0.470、0.330、0.100、0.070`。中间两个区间的三个seed均为负。source-only同一bearing的区间均值`0.330、0.350、0.210、0.080、0.060`，global MMD为`0.390、0.450、0.340、0.130、0.080`。这说明负向主趋势在未对齐分支中已存在，MMD没有修复，conditional MMD还强化了首末均值差的负向幅度。

B3_3 conditional MMD前四区间的平均区间内ρ分别为`−0.320、−0.080、−0.470、−0.280`，末区间为`+0.790`，三个seed的末区间均为正。其五个区间预测均值约为`0.170、0.110、0.000、−0.030、0.240`。末端升高不能抵消前面较长时间的错误排序。这一形态在source-only和global MMD中也出现，提示仅增加对齐正则并没有消除共有失败模式。

## 解释边界

**证据：** B3_2主要是跨大段时间的反向轨迹；B3_3主要是前中段反向与末端跃升叠加；两者都不是单个噪声峰值。三分支共有这些形态，故不能把失败全部归因于conditional MMD。MMD压低了identity probe准确率，却没有保证健康方向正确。

**仍未确定：** 现有预测曲线无法区分是工程特征不含稳定退化信息、训练目标以时间代理代替真实健康状态、bearing间退化阶段不同步，还是模型结构/正则权重不合适。不能仅凭这张图断言某一个原因，更不能据此对B3_4/B3_5作结果预言。

**后续建议：** 下一步先对训练来源与留出bearing的特征/健康代理关系做只读审计，尤其核查B3_2中段和B3_3末端对应的原始工程特征趋势；若要新建方法实验，应先明确健康方向锚点或可靠事件标签，再冻结独立协议。不要简单翻转某个bearing预测符号，不根据本次开发结果调λ或时间区间。

## 统计与完整性检查

诊断输出为27条轨迹摘要、135条区间指标和18条配对变化，来源文件哈希已记录。没有p值或置信区间；每bearing只有三个seed，测量点和时间区间高度相关，不能把48,645个点当独立样本。本次只作描述性诊断，不声称显著性、因果性或外部泛化。11/11类统计谬误已检查：需重点防范总体与分段结论不一致（Simpson式聚合风险）、endpoint伪重复、事后选段/多重比较、将identity下降解释成健康改善、把时间相关当成真实损伤因果。其余类型（生态、Berkson、collider、基率、均值回归、生存者、反向因果）在本次设计中未见直接证据或不适用，但无法仅凭这些CSV完全排除选择偏差。未重跑训练，Verification Status保持ANALYZED。
