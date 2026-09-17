# z=8 AutoEncoder Representation Audit方案（2026-09-17）

## 目的

区分两个可能原因：

1. AutoEncoder latent本身缺少可跨bearing读取的退化信息；
2. latent含有相关信息，但初始位置不对齐或当前Level/Trend/Movement读出过于简单。

## 固定分析

- 冻结现有AutoEncoder和616条latent记录，不重新训练；
- 比较raw z与每个bearing前15点中心化的z；
- 单维度Spearman：normalized lifetime、reconstruction MSE、delta-z；
- 计算PCA方差和effective dimension，检查8维表示实际使用了多少自由度；
- 计算五个bearing从早期到末期的方向向量及两两cosine；
- leave-one-bearing-out三阶段线性分类；
- leave-one-bearing-out连续lifetime Ridge回归；
- chronological bearing-identity线性分类，检查身份/设备特征是否容易读取。

## 边界

early/middle/late来自normalized lifetime三等分，只是弱顺序标签，不是物理损伤阶段。Identity probe的训练和测试都包含相同bearing，因此只用于检测身份信息。当前CSV只有35Hz12kN，不能检验operating-condition信息，必须标记为未测试。

## 描述性判断

- 若中心化显著改善跨bearing阶段/时间探针，主要问题偏向对齐和Interpreter；
- 若raw与centered均差、身份容易预测、方向不一致或effective dimension接近1，Encoder表示目标是主要瓶颈；
- 若跨bearing探针良好但距离Level不稳定，则Interpreter读出定义是主要瓶颈。

阈值仅用于组织证据，不作统计显著性声明。
