# Condition 3 条件 MMD：数值预检设计与执行边界

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-19
- Verification Status: UNVERIFIED（本文件是执行前设计；未计算 MMD 或其梯度）
- Version Label: source_conditional_mmd_numerical_preflight_design_v1
- 上游依据：[三分支草案](xjtu_source_conditional_mmd_protocol_2026-09-19.md)、[缓存覆盖预检](xjtu_source_conditional_mmd_metadata_preflight_2026-09-19.md)、已完成的 Source-DANN 训练记录。

## 要回答的问题

在严格不访问留出 bearing 的情况下，能否构造三分支完全配对的 mini-batch，并选择一个非退化、梯度可控的 MMD 核与正则尺度？这属于**实现与数值可行性门槛**，不是健康效果评价。

## 已有静态证据

- 三个开发 bearing 均可各取 300 个训练 endpoint。B3_1/B3_2 在五段各有60个；B3_3为60/60/59/60/61。
- 原 Source-DANN 的普通随机 batch_size=64 不保证 batch 内同时有两个 bearing 的五个时间区间；直接沿用会使 conditional MMD 的某些区间项缺失。
- 原 source-only 的九个折×seed，在第1 epoch 的健康 MSE 范围为0.204005–0.512681，描述性平均为0.335994。该值只用于识别数量级，**不能用标量 loss 大小代替梯度比例或确定 MMD 权重**。

## 预定配对规则（待代码测试）

每个 batch 从两个**训练** bearing 的每个时间区间各取6个 endpoint，形成 `2 × 5 × 6 = 60` 个样本。每 epoch 设10个 batch；同一 fold、seed、epoch 的三个分支使用完全相同的索引序列。每个 bearing×区间先以固定 seed 洗牌，再按循环索引取足60次。这样B3_3的第三段在该 epoch 有1个重复，第五段有1个未用；洗牌每 epoch 改变，使重复/未用位置轮换。该偏差必须在执行报告中计数，不能隐藏。

在这个设计下，三分支总训练步数相同，但与历史 Source-DANN 的64样本随机 batch 不同。因此**必须重新训练同一采样器下的 source-only 对照**；既有 Source-DANN 只能作为历史参照，不用于严格配对归因。若循环取样的重复被认为不可接受，可另立等量59/区间的新方案，但不能在看到 LOBO 指标之后切换。

## 损失和带宽候选（待数值预检冻结）

- 对两个 source bearing 的16维 embedding 使用 RBF 核的**有偏、非负** MMD² 估计：`mean(Kxx) + mean(Kyy) - 2 mean(Kxy)`，包含对角项；浮点误差允许极小负值，但不能出现实质负值或非有限值。
- Global 分支在每个60样本 batch 的两个30样本 source 集合间计算一项；conditional 分支在五个6对6的时间进度子集分别计算，再对五项等权平均。source-only不加MMD。三分支保留相同健康MSE与初始模型参数。
- 候选核尺度只由该 fold 两个训练 bearing 的初始化 embedding 估计，不接触留出 bearing；global 与 conditional 分支在同一 fold/seed 使用相同的冻结尺度。待检验的多尺度候选为训练来源成对距离中位数的0.5、1、2倍；若中位数接近零，直接停止而不是临时调整到使 LOBO 结果好看的值。
- 正则权重尚未确定。只允许用训练来源的初始化批次计算健康MSE与MMD对 encoder 参数的梯度范数，并依据预先记录的比例规则一次性冻结；不能依据任何留出 bearing 的健康或 identity 指标调权重。

## 数值预检应通过的检查

1. 对每 fold、seed、epoch 的采样索引，验证 batch_size=60、十个 bearing×区间组合各为6、索引均属两个训练 bearing、三个分支索引完全相同；受保护 B3_4/B3_5 路径一旦出现立即失败。
2. 对合成相同分布输入，MMD²接近0；交换两个 source 后不变；对有差异输入为有限非负数；反向传播产生有限梯度。上述仅是数学和实现检查，不证明真实健康语义。
3. 对每个 fold/seed 的**训练来源**初始化表示，记录距离中位数、各核的均值、global/conditional MMD²、健康MSE、以及两种损失对共享 encoder 的梯度范数；不能输出或用于选择留出健康指标。
4. 仅当全部9个 fold×seed 的尺度非退化、损失有限、梯度有限且三分支输入严格配对时，才确定一个全局权重规则并写入**新**冻结配置。任何失败都写入记录并停止，不对失败配置原地改写。

## 当前决策

采样结构和检查标准已明确，但**数值预检尚未执行，核尺度与权重未冻结，正式训练仍不可启动**。下一步是实现独立、只读开发缓存的预检及定向单元测试；执行前应先核验访问保护与输出路径，且不得改动既有 Source-DANN 脚本、配置或结果。

## 局限

时间进度区间不是物理退化阶段；MMD数值稳定也不代表健康排序会改善。三条开发 bearing 的实验单位很少，不能把一个 batch 内的 endpoint 当成独立 bearing。该方案是针对当前项目的受控改写，不是CDAN/DSAN原版复现。
