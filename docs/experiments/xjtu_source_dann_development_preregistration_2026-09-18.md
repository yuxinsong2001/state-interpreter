# XJTU-SY Condition 3 Source-DANN开发实验预注册

## 研究问题

在严格LOBO条件下，仅利用两个训练bearing进行身份对抗，是否能比相同结构的source-only模型产生更一致的健康排序，并把embedding中的bearing identity准确率降低到0.80以下？

## 实验设计

- 开发集：`Bearing3_1`–`Bearing3_3`，三折LOBO。
- 受保护：`Bearing3_4`验证、`Bearing3_5`盲测，均不得读取。
- 输入：65维signed-log1p特征，最多128步因果上下文。
- 模型：单向GRU(32) → 16维embedding → 健康回归头。
- Source-DANN额外使用：16维embedding → GRL → domain discriminator。
- 对照：source-only的GRL系数固定为0；其他条件完全相同。
- 健康目标：normalized lifetime，仅用于两个训练bearing。
- 每个训练bearing等量使用300个endpoint。
- 三个固定seed、30 epochs、固定最终epoch，不用外层测试选择checkpoint。

## 评价门槛

Source-DANN必须同时满足：

1. 三个LOBO测试bearing的平均Spearman均为正；
2. 至少两个bearing平均Spearman不低于0.5；
3. 每个bearing至少两个seed为正；
4. embedding identity probe平均准确率不高于0.80。

Source-only用于配对归因，不单独决定是否打开验证集。只有Source-DANN健康与identity门槛同时通过，才允许读取`Bearing3_4`。

## 停止规则

若失败，不根据结果挑选seed、改变GRL权重、增加epoch或读取保护集。首先分析是健康信息被对齐损坏，还是identity未被有效抑制，再决定是否进入MMD/CORAL或阶段条件对齐。

## 当前状态

协议已写入配置，但必须在核心代码测试、冻结哈希和no-cache-read preflight完成后才可执行。
