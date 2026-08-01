# State Interpreter 正式文献矩阵

检索日期：2026-07-25  
研究对象：基于振动信号或其表征，对旋转机械/轴承的当前健康状态进行解释  
当前边界：优先研究当前状态、退化阶段、连续健康指标和状态转移；RUL 仅作为后续扩展

## 1. 检索问题

本轮文献调研围绕以下问题展开：

1. 文献如何定义 `state`：离散阶段、连续 Health Indicator（HI），还是隐状态？
2. 状态标签从哪里来：人工规则、寿命比例、聚类、变点或故障记录？
3. 模型是否依赖连续的 run-to-failure 序列？
4. 原方法中的 feature extractor 能否替换为冻结的 VibFM `z_health`？
5. 是否存在可复现的公开代码？

## 2. 检索范围与纳入标准

### 主要检索词

- `"bearing degradation stage" vibration autoencoder`
- `"hidden health state" bearing vibration`
- `"health indicator" bearing latent space autoencoder`
- `"condition assessment" bearing hidden Markov model`
- `"early degradation point" bearing health indicator`
- `"degradation stage detection" bearing GitHub`

### 纳入标准

- 输入至少包含振动信号或从振动信号得到的特征；
- 输出能够表达当前健康状态，而不只是故障类型；
- 论文说明了状态构造、阶段划分、HI 或隐状态发现方法；
- 至少能核验论文题目、方法、数据集和主要实验目标。

### 代码状态说明

“未发现作者公开代码”表示截至 2026-07-25，在论文页面、题目检索和 GitHub 检索中没有核验到作者仓库；这不等价于代码一定不存在。

## 3. 迁移等级

- **A—可直接替换**：原方法先提取通用特征，再进行聚类、分类、变点或状态建模；可直接用 VibFM embedding 替代该特征。
- **B—可部分替换**：可以用 VibFM 替换振动分支，但需要额外的时序、多模态、投影或训练适配。
- **C—不能直接替换**：状态表示与原 feature extractor 的训练损失深度绑定；可以借鉴目标函数或状态定义，但不能只换输入。

## 4. 核心文献矩阵

| 文献 | State 定义 | 输入与标签 | 方法 | 数据集 | 主要评价 | 代码 | VibFM 迁移判断 |
|---|---|---|---|---|---|---|---|
| Mannone et al., 2026, VibFM | `z_health` 表示健康相关因素；Paderborn 下游为 Healthy/Inner/Outer 三类诊断状态 | 128×128 log-STFT；自监督预训练，下游使用故障标签 | ViT/MAE 式基础模型，分离 `z_health` 与 `z_nuisance` | 16 个数据集、约 400 h；Paderborn 下游 | 表征学习损失、下游分类与跨域表现 | [官方代码](https://github.com/SFZ-Uni-Stuttgart/VibFM)；未提供 checkpoint | **锚点模型**。提供 representation，不等于完整 State Interpreter；需要状态头或时序模型 |
| Juodelyte et al., 2022 | 将轴承寿命自动划分为若干可执行的 degradation stages | 高频水平/垂直振动；先无标签分段，再生成伪标签 | AutoEncoder latent space → K-means 寿命分段 → 多输入监督分类器 | FEMTO-ST/PRONOSTIA run-to-failure | 自动分段与人工划分的一致性；阶段分类可靠性/准确率；跨轴承实验 | [作者代码](https://github.com/DovileDo/BearingDegradationStageDetection) | **A**。用有序 `z_health` 替换 AE latent，再做 K-means 与分类，是最适合的首个原型 |
| Chen et al., 2023 | 连续、可解释 HI；同时检测 Early Degradation Point（EDP） | 全寿命振动；部分监督信息与退化趋势约束 | 半监督 DCNN AutoEncoder；latent-space variance maximization；辅助 EDP 层 | PHM 2012/FEMTO-ST、NASA IMS | HI 单调性/趋势质量、EDP 检测、与对比 HI 方法比较 | 未发现作者公开代码 | **B/C**。可在 `z_health` 上训练轻量 HI/EDP 头；但其 latent variance 约束与 encoder 联合训练，不能原样直接替换 |
| Alfeo et al., 2024 | Regular、Degraded、Critical 三阶段 | 振动与温度；规则生成阶段标签：持续振动阈值和 RMS 突升决定边界 | 单模态/多模态 AutoEncoder 学习表征，再接树模型等分类器 | PRONOSTIA 的 B11、B12、B21 | 三阶段分类表现；不同融合架构和分类器比较 | 未发现作者公开代码 | **B**。振动 encoder 可换为 `z_health`；温度必须保留独立分支并增加 fusion adapter |
| Cartella & Sahli, 2015 | 单调退化过程中的 CHMM 隐状态；状态数可在线增加 | 连续采集的振动窗口；无预设完整状态标签 | 手工统计/WPT 特征 → KPCA → Change-Point Detection → left-right CHMM；Viterbi 在线推断 | NASA IMS run-to-failure | 新退化状态检测时刻、状态序列合理性；不同特征组合比较 | 未发现作者公开代码 | **A/B**。以 `z_health` 代替统计/WPT 特征，再接 CPD+left-right HMM；需要严格时间顺序 |
| Wang et al., 2021 | 多个 bearing performance degradation states，并定位初始退化 | 原始信号提取的特征；无人工逐窗标签 | SOM 拓扑表示产生退化特征 → HMM 状态评估 | 两组轴承退化实验数据 | 退化指标平滑性、初始退化定位及对比实验 | 未发现作者公开代码 | **A**。可用 `z_health` 替代原始特征/SOM 输入；也可比较是否仍需 SOM 压缩 |
| Qin et al., 2023 | 连续 HI，服务于相似度 RUL 预测 | 振动序列；power-function 型退化标签用于监督约束 | Supervised multi-head self-attention AutoEncoder（SMHSA-AE）→ HI → similarity RUL | 齿轮箱轴承实验数据与公开轴承数据 | HI 质量指标及 RUL 预测误差 | 未发现作者公开代码 | **B/C**。可把 `z_health` 序列送入 attention-HI head；但监督标签和 AE 联合目标无法仅靠替换 extractor 保留 |
| Wu et al., 2024 | 连续 HI，并检测 early fault | 振动包络谱；滑动数值窗口产生初始 HI/伪标签 | CAE 复合损失构造 HI；对比预训练增强早期故障敏感性 | 论文中的轴承全寿命实验 | HI 趋势表现、早期故障检测与对比方法 | 未发现作者公开代码 | **B/C**。可借鉴伪标签和 early-fault head；包络谱 CAE 与其训练损失耦合，不能直接整体替换 |
| Zhou et al., 2016 | 连续 Degradation Value；`DV≤0` 为正常，`DV>0` 表示退化且数值反映程度 | 振动信号；用健康样本建立正常边界 | Lifting Wavelet Packet Symbolic Entropy → SVDD | 轴承全寿命试验数据 | 初始退化检测与 degradation assessment；和 HMM 等方法比较 | 未发现作者公开代码 | **A**。用健康期 `z_health` 训练 One-Class SVM/SVDD，距离作为 HI，是低成本的重要 baseline |
| Singleton et al., 2014 | 振动序列中的 hidden health states 与状态边界 | 时频特征；无逐窗状态标签 | 时频分布特征 → 隐状态/变点发现 | 轴承退化振动数据 | 状态发现与退化过程解释 | 未发现作者公开代码 | **A/B**。可以用有序 `z_health` 替换时频特征；具体复现受正文和实现可得性限制，优先作为概念参考 |

## 5. 逐篇分析

### 5.1 VibFM：State Interpreter 的上游表征

VibFM 把振动信号转成 embedding，并尝试分离健康相关因素与 nuisance factors。它已经解决了“如何表示一个窗口”的一部分，但没有解决以下问题：

- 连续健康程度如何定义；
- 退化阶段和阶段边界如何产生；
- 相邻时间点的状态如何保持一致；
- 何时输出维护决策；
- 如何表达模型的不确定性。

因此，合理架构是：

`振动窗口 → VibFM z_health → State Interpreter → HI/阶段/置信度/状态转移`

### 5.2 Juodelyte et al.：最适合首先迁移

该方法的 feature extractor 与后续状态构造相对解耦。AutoEncoder 只负责把振动降到低维 latent space；K-means 才负责生命周期分段，监督分类器负责新样本阶段识别。

迁移版本：

`按时间排序的振动窗口 → frozen VibFM z_health → 标准化/可选 PCA → K-means 分段 → 阶段分类器`

可行条件：

- 数据必须来自同一轴承的连续 run-to-failure 序列；
- 必须保存 bearing/run ID 和时间戳或采集顺序；
- 聚类后仍需根据时间位置和信号趋势解释每个簇；
- 训练/测试必须按 bearing 划分，不能随机拆窗口。

### 5.3 Chen et al.：适合构造连续 HI

该工作不只做分类，而是让 latent representation 沿退化方向形成可解释变化，并输出 EDP。它与“健康程度是多少、何时开始退化”高度一致。

迁移时不应声称“直接复现 SSALSVM”，因为 variance-maximization loss 会反向训练原 encoder。更合理的两种方式是：

1. 冻结 VibFM，在 `z_health` 上训练小型投影头，输出一维 HI 和 EDP；
2. 后期允许参数高效微调 VibFM，再加入趋势/方差约束。

### 5.4 Alfeo et al.：适合有温度或工况信息时使用

这篇论文说明 `state` 可以由工程规则定义，而不一定需要人工逐点标注。其三阶段定义很适合汇报，但阈值（如振动持续超过某值）是数据集和设备相关的，不能直接搬到新设备。

如果项目只有振动，先不采用多模态版本；如果以后有温度、转速、负载等元数据，可采用：

`z_health + 温度/工况编码 → fusion MLP → 三阶段概率`

### 5.5 Cartella & Sahli：适合发现状态和在线更新

left-right HMM 对状态转移施加“只能保持或向更差状态前进”的结构约束，比普通逐窗口分类更符合单调退化设备。Change-Point Detection 用于估计初始状态数并发现新状态。

迁移版本：

`有序 z_health → 可选降维 → CPD → left-right HMM → Viterbi 当前状态`

它不能用于无顺序的独立诊断样本。若 VibFM 项目最终只提供单次独立测量，则 HMM 路线暂不可行。

### 5.6 Zhou et al.：适合作为最简单的健康边界 baseline

只用健康期 embedding 拟合 One-Class SVM/SVDD，再把到健康边界的有符号距离定义为 HI。这一方案：

- 不需要完整退化标签；
- 不需要复杂时序网络；
- 能快速检查 `z_health` 是否真的含健康信息；
- 但不天然保证跨设备单调、平滑和可校准。

它应当早于 GRU、Attention 等复杂模型实现。

## 6. 横向结论

### 6.1 文献中的 State Interpreter 并非统一术语

相关工作主要分散在以下术语下：

- health indicator construction；
- degradation stage detection/recognition；
- condition assessment；
- hidden health-state discovery；
- early degradation/fault detection；
- degradation modeling。

因此，老师所说的“理论研究较少”可以理解为：目前缺少一个统一框架，把 foundation-model embedding 系统地变成可解释、连续、带不确定性且可用于维护决策的状态；但各个子问题已有可迁移方法。

### 6.2 三类输出可以共存

一个完整 State Interpreter 可以同时输出：

- 连续值：`health_score ∈ [0,1]`；
- 离散阶段：Healthy / Early Degradation / Severe Degradation / Failure；
- 事件：EDP、change point、状态转移；
- 置信度：当前判断的不确定性。

连续 HI 可用于趋势，离散阶段可用于汇报和决策，变点用于告警；三者不是互斥选项。

### 6.3 VibFM 最适合替代“通用特征提取器”

最容易迁移的是：

- AutoEncoder latent → K-means/分类器；
- 手工统计或 WPT 特征 → CPD/HMM；
- 手工健康特征 → SVDD/One-Class SVM。

不应直接宣称可无缝迁移的是：

- encoder 与单调性、方差、重建或监督退化标签联合训练的方法；
- 同时依赖温度等传感器的多模态方法；
- 必须依赖连续时间序列、而当前数据可能只有独立测量的方法。

## 7. 建议的实现优先级

### P0：先确认数据与 checkpoint

1. 向老师申请 pretrained VibFM checkpoint；
2. 确认第一目标数据是 run-to-failure 序列还是独立测量；
3. 确认是否能获得 unit ID、时间顺序、温度、转速和负载。

### P1：最小可行实验

使用一个公开 run-to-failure 数据集（优先 FEMTO-ST/PRONOSTIA）：

1. frozen VibFM 提取 `z_health`；
2. 健康期 One-Class SVM/SVDD 距离作为连续 HI；
3. `z_health + K-means` 自动生成退化阶段；
4. 使用 Logistic Regression/MLP 预测阶段；
5. 按 bearing 划分训练、验证和测试。

### P2：加入时间约束

1. Change-Point Detection；
2. left-right HMM；
3. 比较是否减少不合理的逆向状态跳转；
4. 比较 EDP 检测时间和阶段 Macro F1。

### P3：复杂模型

只有当简单方法不足时，再比较：

- GRU；
- GRU + Attention；
- temporal Transformer；
- uncertainty-aware state head。

GRU 和 Attention 是候选实现，不是项目的先验要求。

## 8. 首轮评价协议

### 连续 HI

- monotonicity；
- trendability；
- robustness；
- 与寿命比例、RUL 或真实 damage 的 Spearman 相关；
- 跨轴承的一致性。

### 离散阶段

- Macro F1、Balanced Accuracy；
- 每阶段 Precision/Recall；
- confusion matrix；
- 状态边界误差；
- 不合理逆向转移次数。

### 早期退化检测

- EDP 检测提前量/延迟；
- false alarm rate；
- 检测后状态是否持续；
- 跨 bearing 泛化。

## 9. 当前可行性判断

### 现在即可完成

- 复现 Juodelyte 论文的阶段生成流程；
- 用公开数据和普通 AutoEncoder 建立基准；
- 完成数据时序、标签生成和评价代码；
- 用手工/AE 特征验证 HMM、K-means、One-Class SVM。

### 获得 checkpoint 后可完成

- 将 AE/手工特征替换为 VibFM `z_health`；
- 检验 VibFM embedding 是否形成退化轨迹；
- 比较 `z_health`、`z_nuisance` 和拼接表征；
- 冻结 VibFM 训练轻量 State Interpreter。

### 需要进一步条件

- 多模态方案需要温度或工况信息；
- 在线状态更新需要真实连续采集或可模拟的数据流；
- 维护决策阈值需要成本、风险或老师给出的工程规则；
- 若要微调 VibFM，可能需要学校服务器。

## 10. 推荐结论

第一版不应直接从 GRU + Attention 开始。最有证据、风险最低的方案是：

`frozen VibFM z_health → One-Class HI + K-means degradation stages → 轻量分类器`

然后增加：

`Change-Point Detection + left-right HMM`

这一顺序能分别回答：

1. `z_health` 是否包含退化信息？
2. 是否能自动形成可解释阶段？
3. 时间约束是否让状态更稳定？
4. 复杂时序网络是否真的带来额外价值？

## 11. 主要来源

- Mannone, G., Fischer, S., & Dazer, M. (2026). *Learning the Language of Vibration: A Self-Supervised Transformer Foundation Model for PHM*. [论文页面](https://papers.phmsociety.org/index.php/phme/article/view/4912)
- Juodelyte, D., Cheplygina, V., Graversen, T., & Bonnet, P. (2022). *Predicting Bearings' Degradation Stages for Predictive Maintenance in the Pharmaceutical Industry*. [arXiv](https://arxiv.org/abs/2203.03259)
- Chen et al. (2023). *An interpretable health indicator for bearing condition monitoring based on semi-supervised autoencoder latent space variance maximization*. [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/acf515/meta)
- Alfeo, A. L., Cimino, M. G. C. A., & Gagliardi, F. (2024). *Recognizing Bearings' Degradation Stage Using Multimodal Autoencoder to Learn Features from Different Time Series*. [Springer](https://link.springer.com/article/10.1007/s42979-024-02635-5)
- Cartella, F., & Sahli, H. (2015). *Online Adaptive Bearings Condition Assessment Using Continuous Hidden Markov Models*. [全文](https://journals.sagepub.com/doi/full/10.1155/2014/758785)
- Wang, D. et al. (2021). *Bearing performance degradation assessment based on topological representation and hidden Markov model*. [SAGE](https://journals.sagepub.com/doi/abs/10.1177/1077546320946633)
- Qin, Y. et al. (2023). *A new supervised multi-head self-attention autoencoder for health indicator construction and similarity-based machinery RUL prediction*. [DOI](https://doi.org/10.1016/j.aei.2023.101973)
- Wu, D., Chen, D., & Yu, G. (2024). *New Health Indicator Construction and Fault Detection Network for Rolling Bearings via Convolutional Auto-Encoder and Contrast Learning*. [MDPI](https://www.mdpi.com/2075-1702/12/6/362)
- Zhou et al. (2016). *Bearing Performance Degradation Assessment Using Lifting Wavelet Packet Symbolic Entropy and SVDD*. [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1155/2016/3086454)
- Singleton et al. (2014). *Discovering the hidden health states in bearing vibration signals for fault prognosis*. [IEEE](https://ieeexplore.ieee.org/abstract/document/7049008/)

