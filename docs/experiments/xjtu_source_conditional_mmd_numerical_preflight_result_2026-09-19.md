# Condition 3 条件 MMD 数值预检结果

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + descriptive validation
- Origin Date: 2026-09-19
- Verification Status: ANALYZED（预检命令完成且定向/全量测试通过；尚未独立复跑，**不是**方法有效性验证）
- Version Label: source_conditional_mmd_numerical_preflight_v1
- 输入：既有 Condition 3 开发缓存 B3_1–B3_3，以及冻结 Source-DANN 配置 SHA-256 `b0b594a60e04366fefbb8894011ad587eb2e7b1749bdb58d1b6df6baae6cc45d`。
- 输出：[diagnostics.json](../../results/2026-09-19_source_conditional_mmd_numerical_preflight_v1/diagnostics.json)，SHA-256 `44c0df4d58cec0ae152d38492caa5e9116a13ed12ee8907767d5eea3b1ee7164`。
- 边界：只对两个训练 bearing 的**初始化**encoder做前向与梯度检查；未训练、未选择checkpoint、未访问 B3_4/B3_5。

## 实现与范围

新增独立的配对采样/MMD数学模块及只读数值预检入口，未修改历史 Source-DANN 训练脚本、配置或结果。每个fold×seed取两个训练bearing，各按5个时间进度区间采样；每批每个bearing×区间取6个endpoint，共60个。预检计算初始16维embedding的成对距离中位数，并用其0.5、1、2倍核尺度的平均RBF MMD²，分别做全局与区间条件化比较。只检查第一批，共3折×3 seed=9组；不声称已经覆盖训练过程中的后续batch或参数更新后尺度。

## 数值结果（9组描述性范围）

| 指标 | 最小 | 中位 | 最大 |
|---|---:|---:|---:|
| 初始化健康MSE | 0.340367 | 0.452752 | 0.790753 |
| embedding成对距离平方中位数 | 0.742648 | 1.219139 | 1.821636 |
| global MMD² | 0.488020 | 0.650662 | 0.818248 |
| conditional MMD² | 0.761762 | 0.858566 | 0.951012 |
| global MMD / 健康MSE 的共享encoder梯度范数比 | 1.479654 | 2.494941 | 3.215883 |
| conditional MMD / 健康MSE 的共享encoder梯度范数比 | 1.495763 | 2.648271 | 3.234230 |

所有9组的核尺度、损失及梯度均有限且非退化，没有触发预检程序的停止条件。由于 conditional MMD 每个子项只有6对6，而 global MMD是30对30，**两者的有偏MMD²原始值不能直接比较优劣**；本预检也不判断健康排序是否改善。上述梯度范数比只是初始化首批的数量级，不能据此直接宣称正则权重已经最优。

## 安全与复现检查

- `pytest -q tests/test_conditional_mmd_preflight.py -p no:cacheprovider`：5 passed。
- `pytest -q -p no:cacheprovider --basetemp _pytest_conditional_mmd_20260919`：194 passed。
- 程序有精确确认令牌、开发bearing及三折检查、缓存路径和哈希检查、输出不覆盖检查；报告列明保护bearing未读取及未训练。
- 预检模块 SHA-256 `b9ce82f1ff014e3bfe1f991df1754b7ce6e5fd4b669364ea45335764dd3be6d5`；入口脚本 SHA-256 `aecca913c890046e33d39a6d27b58a7d3f3951c5176581e9d294205fb667d543`。

## 结论与下一门槛

**数值预检通过，但三分支开发训练尚未获准。**下一步先确定唯一正则权重规则、完成三分支严格同索引训练实现及对应安全测试，再将这些新代码与新配置独立冻结；不能改写 Source-DANN 冻结工件。训练完成后再按既有健康排序与identity双门槛判断，不能凭当前MMD数值开启B3_4。

## 解释限制

九组是三个开发fold与三个seed，不是九个独立bearing；同一bearing的endpoint具有时间相关性。五段 normalized lifetime 不是物理损伤阶段。初始化数值稳定仅说明代码路径可计算，不能证明跨bearing泛化，也不能用于统计显著性推断。
