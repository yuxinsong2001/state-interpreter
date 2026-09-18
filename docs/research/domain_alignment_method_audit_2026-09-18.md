# 跨 bearing 表示对齐方法审计

## 当前问题

Feature LSTM 与紧凑 TS2Vec 的结果共同表明：65维输入以及学习后的表示仍强烈保留 bearing identity，而且不同 bearing 的健康方向并不一致。下一方法必须直接处理“身份信息过强”问题，而不是只增加时间模型复杂度。

## 候选方法

### DANN / gradient reversal

DANN在主任务之外加入domain discriminator。Gradient Reversal Layer在前向传播中保持恒等，在反向传播中反转domain loss的梯度，使encoder学习对主任务有用、但难以识别domain的表示。

优点：目标与当前identity问题直接一致；可与健康回归联合训练；PyTorch成熟库已有清楚实现。风险：如果健康进度与bearing身份相关，过强对齐可能删除健康信息。

### MMD / DAN

MMD直接缩小不同domain表示分布的核均值差异。它不需要对抗训练，通常更稳定。

风险：全局边缘分布对齐可能把不同健康阶段错误地对齐，尤其当不同bearing的生命周期采样密度不同。当前尚无可靠阶段标签，因此不优先。

### CORAL

CORAL对齐表示的二阶统计量，结构简单，也适合做后续对照。

风险：只对齐均值/协方差不足以保证健康方向一致；作为第一候选针对性弱于直接identity adversary。

## 决策

第一步采用DANN-inspired source-domain adversarial learning，但不声称执行标准DANN target adaptation：每个LOBO fold只使用两个训练bearing，将它们作为两个source domains；被留出的第三个bearing完全不参与训练。

实验同时训练两个完全配对的分支：source-only和source-DANN。两者模型、健康监督、样本、seed与训练轮数相同，唯一差异是gradient reversal系数。这样可以直接判断identity抑制是否真正改善跨bearing健康表示。

## 主要依据

- [Ganin et al., *Domain-Adversarial Training of Neural Networks*, JMLR 17(59), 2016](https://www.jmlr.org/papers/v17/15-239.html)：DANN与Gradient Reversal原始方法。
- [Jiang et al., *Transfer-Learning-Library*](https://github.com/thuml/Transfer-Learning-Library)：MIT许可的成熟PyTorch实现，覆盖DANN、DAN与domain generalization方法。
- [Zhao & Liu, *Cross-condition and cross-platform remaining useful life estimation via adversarial-based domain adaptation*, Scientific Reports, 2022](https://www.nature.com/articles/s41598-021-03835-2)：在bearing RUL跨工况问题中使用对抗域适配。

## 限制

标准DANN通常在训练时使用未标注目标域数据。本项目为了保持严格LOBO，不允许使用被留出bearing，因此当前设计属于DANN启发的source-domain invariance/domain generalization。即使结果改善，也不能直接推广为标准unsupervised domain adaptation结论。
