# z=8 AutoEncoder Representation Audit结果（2026-09-17）

## 核心判断

证据不支持“当前AutoEncoder完全没有可迁移的退化信息”。相反，latent中的时间排序信息很强、五个bearing的早期到末期方向高度一致、bearing身份只略高于随机水平。

更准确的问题是：8维latent几乎坍缩为一个一维公共变化轴，而不同bearing在该轴上的起点、速度和绝对阶段刻度不一致。当前瓶颈更偏向状态标定与Interpreter读出，而不是立即重训AutoEncoder；但一维表示也限制了未来区分不同健康因素、故障模式和工况的能力。

## 数据与方法

- 冻结现有AutoEncoder与616条35Hz12kN latent记录；
- 不修改模型、不重新训练；
- 比较raw z与每个bearing前15点中心化的z；
- 五折leave-one-bearing-out阶段分类和lifetime Ridge探针；
- PCA effective dimension、bearing identity、退化方向cosine和单维度相关性；
- early/middle/late是normalized lifetime三等分弱标签，不是物理损伤阶段。

## 主要结果

| 指标 | Raw z | Early-centered z |
|---|---:|---:|
| Effective dimension | 1.0016 | 1.0012 |
| PC1解释方差 | 99.9203% | 99.9380% |
| LOBO阶段balanced accuracy均值 | 0.623 | 0.616 |
| LOBO阶段最差fold | 0.333 | 0.350 |
| LOBO lifetime Spearman均值 | 0.884 | 0.789 |
| LOBO lifetime Spearman最差fold | 0.699 | 0.670 |
| LOBO lifetime MAE均值 | 0.248 | 0.211 |
| Bearing identity balanced accuracy | 0.302 | 0.222 |

五个bearing的十个非对角early-to-late方向cosine范围为0.9944–0.9999，中位数0.9993。

## 解释

### 1. AutoEncoder确实学到了公共退化方向

- raw z的跨bearing lifetime Spearman每折均为正，最差仍为0.699；
- 所有8个维度在五个bearing上的相关符号一致；
- 五个early-to-late方向几乎完全平行；
- centered identity balanced accuracy为0.222，接近五分类chance 0.2。

因此，现有latent并非主要编码bearing身份，也不是五个bearing各走各的随机方向。

### 2. 8维表示实质上只有一维

PC1解释超过99.9%的训练bearing方差，effective dimension约1.00。八个坐标基本是在重复表达同一条轴。这解释了为什么不同维度与lifetime的相关性几乎同步，也说明增加复杂GRU或Attention无法创造latent中不存在的独立健康因素。

一维并不等于无效：它可以作为健康指标候选。但它不足以证明已经分离损伤、噪声、故障模式和工况。

### 3. B1_4主要是绝对标定失败，不是排序信息消失

B1_4的raw lifetime Spearman为0.855，说明线性探针仍能得到正确时间排序；但连续预测MAE为0.506、R²为-2.563，三阶段balanced accuracy只有chance 0.333。即顺序存在，但从latent位置映射到统一阶段/寿命刻度失败。

前15点中心化把B1_4 MAE改善为0.292，却没有稳定解决阶段分类，且Spearman降到0.721。这说明初始偏移是问题之一，但单纯中心化不是完整方案。

### 4. Level/Trend/Movement的定位

- Level距离能够利用这条一维退化轴，但欧氏距离丢失符号，且不同bearing缺少统一尺度；
- Trend继承Level标定和窗口问题；
- Movement能够检测轴上的短期移动，但仍不能区分退化、噪声与异常。

所以当前主要瓶颈是“如何把共同排序轴校准为有语义的state”，不是证明latent完全无信息。

## 对原问题的回答

| 候选原因 | 当前证据判断 |
|---|---|
| AutoEncoder没有退化信息 | 不支持 |
| AutoEncoder表示过于单一 | 支持，effective dimension≈1 |
| Bearing身份污染是主要问题 | 不支持，中心化后接近chance |
| 不同bearing退化方向不一致 | 不支持，direction cosine接近1 |
| Level/阶段映射标定不足 | 支持，尤其B1_4排序好但绝对误差和阶段分类差 |
| 单纯中心化即可解决 | 不支持，改善不一致 |

综合结论是`mixed_or_inconclusive_bottleneck`，但证据重心偏向Interpreter标定，同时保留Encoder表达能力过窄这一限制。

## 下一步

建立最小的signed degradation-axis Interpreter：

1. 只用训练bearing学习共同PC1/early-to-late方向；
2. 用训练bearing时间顺序确定方向正负；
3. 新bearing只用前15点建立自身参考；
4. 输出signed projection，而不是无符号欧氏距离；
5. 与当前Level、Residual GRU和阶段探针比较；
6. 不使用B1_4/B1_5重新选择方向或参数。

如果signed axis只能排序但仍不能产生可迁移阶段刻度，则下一瓶颈是缺少物理健康标签/统一标定，而不是继续增加网络复杂度。

## 推断边界

11类fallacy检查已覆盖。本实验只有一个工况，无法检查condition invariance；所有bearing均已查看；LOBO五折不是设备总体的显著性样本；弱时间标签不能被称为真实损伤阶段；线性探针也可能低估非线性信息。因此Verification Status为ANALYZED，而不是新的盲测确认。
