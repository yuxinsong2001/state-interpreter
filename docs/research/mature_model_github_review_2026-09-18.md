# 成熟开源模型检索：State Interpreter替代路线（2026-09-18）

## 检索问题

当前`STFT → 重建式AutoEncoder → z=8 → 单一Level轴`是否过于简单？公开实现如何处理健康表征、时间依赖、跨工况迁移与阶段转换？

## 结论摘要

答案是“部分是”。当前模型容量较小，但更关键的问题不是层数，而是目标函数和系统分解：纯重建不保证健康语义；单测量编码不利用退化上下文；单一线性轴假定所有bearing方向一致；没有显式处理工况域偏移和退化起点。成熟路线通常组合“更有任务意义的输入/表征 + 时序模型 + 阶段检测或域适配”，而不是只把AutoEncoder做大。

## 候选开源实现

### 1. XJTU-SY Bearing RUL Benchmark

- GitHub: https://github.com/thfmn/xjtu-sy-bearing
- 直接使用XJTU-SY，提供15-fold LOBO、566项测试、六类模型和两阶段onset pipeline。
- 方法包括65维统计/频域特征、LightGBM、1D-CNN、2D-CNN、TCN-LSTM、CWT+Transformer/MLP。
- 仓库报告的最佳LOBO模型是约5,793参数的Feature BiLSTM，而非最大模型；流程先用RMS、kurtosis、CUSUM等检测退化起点，再在退化区间预测。
- 对当前项目的价值：最高。可复用其特征提取、LOBO协议和onset detection作为强基线，但其目标是RUL，不能直接把输出称为RL state。

### 2. rul-adapt

- GitHub: https://github.com/tilman151/rul-adapt
- 面向RUL的PyTorch Lightning域适配库，包含LSTM-DANN、ADARUL、LatentAlign、TBiGRU、Consistency-DANN、Conditional MMD等，并标注支持XJTU/FEMTO。
- 核心思想：同时学习退化任务与“无法辨认工况”的表示，或使用MMD/latent alignment缩小源域和目标域分布差异。
- 对当前项目的价值：用于解决Condition 1→3分布偏移，而非直接替代State Interpreter。应在建立可靠健康目标后再加入。

### 3. TS2Vec

- GitHub: https://github.com/zhihanyue/ts2vec
- AAAI 2022官方实现，使用多尺度、时间级和实例级对比学习；支持因果滑窗推理并输出逐时间戳表示。
- 相比重建式AutoEncoder，它要求同一序列不同上下文视图产生一致表示，并在多个时间尺度保留时序结构。
- 对当前项目的价值：适合作为Encoder替代候选；需要把连续若干测量组织成序列，不能继续把每个CSV完全独立编码。

### 4. TS-TCC

- GitHub: https://github.com/emadeldeen24/TS-TCC
- IJCAI 2021官方MIT实现，使用强/弱增强、跨视图未来预测以及contextual contrastive loss。
- 论文报告了跨域时序实验，强调时间特征和增强不变性。
- 对当前项目的价值：适合学习对振动扰动稳健的时序embedding；但原任务主要是分类，必须重新定义适合轴承信号的增强和health readout。

### 5. AdaTime

- GitHub: https://github.com/emadeldeen24/AdaTime
- TKDD 2023官方MIT基准，统一比较DANN、Deep CORAL、MMD、CDAN、CoDATS、CoTMix等11种时序域适配方法，并明确提供source-only/target-only上下界和多随机种子输出。
- 对当前项目的价值：不是单个State Interpreter，而是正确比较跨工况方法的实验框架。尤其提醒不能使用目标域标签选择超参数。

### 6. TSLANet

- GitHub: https://github.com/emadeldeen24/TSLANet
- ICML 2024官方MIT实现，使用adaptive spectral block、interactive convolution和自监督masking，目标是同时建模长短期关系并降低噪声敏感性。
- 对当前项目的价值：可作为较现代、轻量的非线性时序Encoder候选；其复杂度和适配成本高于TS2Vec/Feature LSTM，不应作为第一复现对象。

### 7. PPDM Framework

- GitHub: https://github.com/panoskom/PPDM_framework
- 将AutoEncoder、health indicator、随机RUL与维护RL明确拆成可替换模块。
- 对当前项目的价值主要是系统架构：支持“表征不等于状态，状态再服务决策”的边界；但其开源描述不能证明其中AutoEncoder能解决XJTU-SY跨工况问题。

## 推荐路线

第一优先不是复制一个大型模型，而是复用XJTU-SY benchmark的两项成熟做法：65维可解释特征和两阶段onset detection，并把Feature LSTM作为强基线。它能直接检验B3_3是否需要阶段分解，也比继续凭经验设计新latent轴更可控。

第二优先使用TS2Vec替换重建式AutoEncoder，输出有时间上下文的embedding；保持同一State Interpreter评价接口，与Feature LSTM及当前模型做LOBO比较。

第三优先在前两者确认健康信号后，引入rul-adapt或AdaTime中的DANN/MMD做跨工况对齐。若基础健康表示本身不可靠，域适配只会对齐错误表示。

暂不优先：直接上大型Transformer、Attention或HMM。Attention不能自动产生健康语义；HMM只能组织已有观测，不能修复观测表示；大型模型在15条bearing上更容易过拟合。

## 后续研究决策原则

本次检索确立了新的默认流程：实验失败后先从已检索的成熟方法族中寻找对应解决方案并复现，再决定是否自行设计模型。自研方法必须建立在成熟基线仍未解决的明确缺口上，并且一次只改变一个关键因素。具体规范见`research_methodology_and_reuse_policy.md`。

## 建议的下一实验

```text
XJTU-SY 65维统计/频域特征
→ onset detector（RMS/kurtosis/CUSUM）
→ 10步Feature LSTM
→ [phase probability, relative progress, transition confidence]
→ LOBO评价
```

该实验应先在B3_1–B3_4开发证据上运行，B3_5继续保留。若该强基线仍无法处理B3_3，再复现TS2Vec；不要同时更换输入、Encoder、Interpreter和评价方式。

## 局限

- `thfmn/xjtu-sy-bearing`是工程化基准，其中多个模型是仓库作者的原创实现或受论文启发，并非所有部分都有同行评审对应物。
- TS2Vec、TS-TCC、AdaTime和TSLANet是通用时序方法，不是为轴承State Interpreter直接设计。
- GitHub stars、测试数量和README结果不能代替本项目上的复现。
