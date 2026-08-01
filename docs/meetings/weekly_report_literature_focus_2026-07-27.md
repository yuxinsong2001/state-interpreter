# 组会汇报重点版：文献中的 State Interpreter 实现方式

汇报日期：2026-07-27

## 1. 本周研究问题

老师在黑板上提出的核心问题可以概括为：

```text
VibFM embedding
      ↓
如何解释 embedding space？
      ↓
如何得到 state？
      ↓
这个 state 是否已经 decision-ready？
```

当前研究边界已经明确：

- 固定使用 VibFM，不再比较 encoder；
- `z_health` 是 State Interpreter 的输入，不一定是最终 RL state；
- 主要调研文献如何把振动特征或 latent embedding 转换为健康状态；
- 最后再判断哪些方法可以迁移到 VibFM。

## 2. 文献调研的总体发现

“State Interpreter”不是 PHM 领域统一使用的关键词。相关实现分散在：

- health indicator construction；
- degradation stage detection；
- condition assessment；
- hidden health-state discovery；
- early degradation point detection；
- RL state representation。

现有文献主要解决 State Interpreter 的局部功能，例如连续 HI、离散退化阶段、隐状态或变点。较少有工作把 foundation-model embedding、可解释健康状态、不确定性、状态转移和 RL 维护决策放入一个统一框架。

因此，可以从文献中归纳出以下六种代表性实现方式。

## 3. 实现方式一：健康边界距离作为连续 HI

代表文献：

Zhou et al. (2016), *Bearing Performance Degradation Assessment Using Lifting Wavelet Packet Symbolic Entropy and SVDD*.

### 原方法

```text
healthy vibration
      ↓
handcrafted degradation features
      ↓
SVDD / one-class model
      ↓
distance to healthy boundary
      ↓
continuous degradation value
```

模型只使用健康期数据建立“正常区域”。新样本距离健康区域越远，退化程度越高。

### 输出 state

```text
health indicator / degradation value
```

### 迁移到 VibFM

```text
healthy z_health
      ↓
One-Class SVM / SVDD
      ↓
distance score
      ↓
HI
```

### 优点

- 不需要完整退化标签；
- 实现简单；
- 可以首先检查 `z_health` 是否真的含有健康信息；
- 适合作为最低成本 baseline。

### 局限

- 距离不一定天然单调；
- 不直接提供退化阶段；
- 对跨设备和工况变化可能敏感；
- 还不是完整的 RL state。

## 4. 实现方式二：Latent clustering 生成退化阶段

代表文献：

Juodelyte et al. (2022), *Predicting Bearings' Degradation Stages for Predictive Maintenance in the Pharmaceutical Industry*.

### 原方法

```text
horizontal/vertical vibration
      ↓
FFT
      ↓
two AutoEncoders
      ↓
16-dimensional latent representation
      ↓
K-means with 3 clusters
      ↓
clusters ordered by average lifecycle position
      +
AE reconstruction-error anomaly threshold
      ↓
4 degradation stages
      ↓
supervised classifier
```

源码显示，四阶段不是直接来自四类 K-means，而是：

```text
3 latent clusters + 1 reconstruction-error anomaly stage
```

### 输出 state

```text
Healthy / Stage 1 / Stage 2 / Stage 3 probabilities
```

### 迁移到 VibFM

```text
ordered z_health
      ↓
optional normalization/PCA
      ↓
K-means
      ↓
clusters ordered by time
      ↓
stage classifier
```

### 优点

- 有作者公开代码；
- 不需要逐窗口人工标签；
- 与黑板上的 Healthy/Damage embedding-space 聚类直接对应；
- 可以输出易于解释的阶段。

### 局限

- 必须有连续 run-to-failure 序列；
- “越晚出现越严重”是人为时间先验；
- 聚类不等于物理真值；
- 原论文部分评价依赖自动生成的标签，存在循环性。

### 本项目中的定位

这是最适合首先迁移到 `z_health` 的离散状态方法，但应该把输出称为 pseudo-label stages，而不是绝对真实状态。

## 5. 实现方式三：变点检测 + Left-right HMM

代表文献：

Cartella and Sahli (2015), *Online Adaptive Bearings Condition Assessment Using Continuous Hidden Markov Models*。

相关参考：

Wang et al. (2021), *Bearing performance degradation assessment based on topological representation and hidden Markov model*。

### 原方法

```text
ordered vibration features
      ↓
optional dimensionality reduction
      ↓
change-point detection
      ↓
left-right HMM / CHMM
      ↓
Viterbi hidden-state inference
```

Left-right HMM 通常只允许：

```text
保持当前状态
或
向更严重状态转移
```

### 输出 state

```text
hidden degradation state
+ state-transition probability
```

### 迁移到 VibFM

```text
ordered z_health
      ↓
change-point detection
      ↓
left-right HMM
      ↓
temporally consistent state
```

### 优点

- 直接建模时间转移；
- 减少 `Critical → Healthy → Critical` 等不合理跳转；
- 状态序列比逐窗口分类更稳定；
- 可用于 early degradation/change-point detection。

### 局限

- 必须有严格时间顺序；
- 状态数和转移假设可能需要人工设定；
- 隐状态仍需要通过 damage、RMS、寿命位置等赋予工程语义；
- 不适用于无关联的单次诊断样本。

### 本项目中的定位

适合作为第二阶段：先产生 HI 或阶段，再用 HMM 加入时间一致性。它比直接使用 GRU 更容易解释。

## 6. 实现方式四：监督学习连续 HI + Early Degradation Point

代表文献：

Chen et al. (2023), *An interpretable health indicator for bearing condition monitoring based on semi-supervised autoencoder latent space variance maximization*。

### 原方法

```text
run-to-failure vibration
      ↓
semi-supervised AutoEncoder
      ↓
latent-space trend/variance constraint
      ↓
continuous HI
      +
early degradation point
```

其目标不是只把样本分成几类，而是让 latent representation 沿退化方向形成可解释变化。

### 输出 state

```text
continuous health indicator
+ early degradation event
```

### 迁移到 VibFM

第一阶段冻结 VibFM：

```text
z_health
      ↓
small regression/projection head
      ↓
HI + EDP probability
```

### 优点

- 能回答“退化程度是多少”；
- 比离散阶段保留更多连续信息；
- 可以检测何时开始退化；
- HI 可用于趋势和维护阈值。

### 局限

- 原论文损失与 AutoEncoder 联合训练；
- 不能声称只替换输入就完整复现原方法；
- HI target 或趋势约束必须先定义；
- 单一 HI 可能无法区分不同故障模式。

### 本项目中的定位

可作为第一版 MLP 的连续输出 head，与 stage probabilities 同时训练。

## 7. 实现方式五：多模态/Hybrid State

代表文献：

Alfeo et al. (2024), *Recognizing Bearings' Degradation Stage Using Multimodal Autoencoder to Learn Features from Different Time Series*。

### 原方法

```text
vibration representation
        +
temperature / other time series
        ↓
multimodal encoding/fusion
        ↓
Regular / Degraded / Critical
```

阶段边界由持续振动阈值、RMS 突升等工程规则产生。

### 输出 state

```text
multimodal degradation-stage probabilities
```

### 迁移到 VibFM

```text
z_health
    +
temperature / load / speed / previous action
    ↓
fusion MLP
    ↓
compact hybrid state
```

### 优点

- 能区分健康变化和工况变化；
- 更符合 RL dynamics，因为负载和 action 会影响未来退化；
- 可以加入维护历史。

### 局限

- 需要同步元数据；
- 工程阈值不能直接跨设备复制；
- 容易学习机器身份或数据集差异；
- 必须区分 nuisance condition 与真正影响 reward/dynamics 的 condition。

### 本项目中的定位

只有在获得可靠的转速、负载、温度或 action history 后才启用。

## 8. 实现方式六：Reward/Transition Predictive State

代表理论：

- Lesort et al. (2018), State Representation Learning for Control；
- Gelada et al. (2019), DeepMDP；
- Zhang et al. (2021), Deep Bisimulation for Control；
- Ni et al. (2024), self-predictive state/history representation。

### 核心思想

健康分类准确并不代表 state 适合 RL。State 应保留预测 reward 和 action-conditioned transition 所需的信息：

```text
s_t, a_t → reward_t
s_t, a_t → s_(t+1)
```

### 可能实现

```text
z_health/history
      ↓
State Interpreter
      ├── HI head
      ├── stage head
      ├── reward prediction head
      └── next-state prediction head
              ↓
compact decision state
```

训练目标可以写为：

```text
L = λ_health L_health
  + λ_stage L_stage
  + λ_reward L_reward
  + λ_transition L_transition
```

### 优点

- 直接检查 state 是否保留决策相关信息；
- 能发现只在健康程度上相似、但 action 后果不同的 state aliasing；
- 与老师黑板上的“Decision ready?”直接对应。

### 局限

- 需要明确的 RL reward、action 和 transition；
- 公开 run-to-failure 数据通常没有维护 action；
- 最终需要 Gearbox/Maintenance Arena 或其他交互环境；
- 比 HI/stage baseline 更复杂。

### 本项目中的定位

它是最终完整 State Interpreter 的方向，不是没有 checkpoint 时首先实现的模型。

## 9. 六种方法的横向比较

| 方法 | 主要输入 | 输出 state | 是否需要序列 | 标签需求 | VibFM 迁移难度 | 推荐阶段 |
|---|---|---|---:|---|---|---|
| One-Class/SVDD | 健康期 embedding | 连续 HI | 否 | 只需健康期 | 低 | 第一阶段 |
| K-means stages | 全寿命 embedding | 离散阶段概率 | 是 | 无逐窗标签，需时间顺序 | 低 | 第一阶段 |
| CPD + left-right HMM | embedding 序列 | 隐状态/转移 | 是 | 可无监督 | 中 | 第二阶段 |
| HI + EDP head | embedding | HI + 退化起点 | 最好有 | HI/弱监督/趋势先验 | 中 | 第一至二阶段 |
| Hybrid fusion | embedding + 工况/action | 融合状态 | 视设计而定 | 状态或决策监督 | 中 | 有元数据后 |
| Predictive state | state/history + action | decision state | 是 | reward + transition | 高 | RL 阶段 |

## 10. 推荐给当前 VibFM 项目的实现路线

### Baseline A：连续健康边界

```text
frozen z_health → One-Class SVM/SVDD distance → HI
```

回答：`z_health` 是否包含可测量的健康偏离？

### Baseline B：离散退化阶段

```text
ordered z_health → K-means → time-ordered stages → classifier
```

回答：embedding space 是否形成可解释阶段？

### Baseline C：单窗口联合输出

```text
z_health → MLP → HI + stage probabilities
```

回答：一个 embedding 是否已经足够？

### Temporal extension

```text
HI/stage sequence → CPD + left-right HMM
```

回答：时间约束能否减少不合理状态跳转？

### GRU/Attention extension

```text
z_health history → GRU/TCN/Attention → state
```

只有当 history-sufficiency test 显示历史显著改善 reward/transition prediction 时启用。

### Decision-ready extension

```text
state + action → reward/next-state prediction → RL evaluation
```

回答：state 是否真正帮助维护决策？

## 11. 推荐汇报结构

1. 研究问题：固定 VibFM 后，如何从 `z_health` 得到 RL state；
2. 文献检索发现：State Interpreter 分散在 HI、stage、HMM、EDP 和 RL representation 等方向；
3. 重点讲上述六种实现方式；
4. 给出横向比较和逐步实现路线；
5. 最后用一页说明本周的软件准备：独立仓库、数据契约、adapter 和六项测试；
6. 请老师确认 checkpoint、第一数据集和第一版 state 输出。

## 12. 5 分钟口头汇报稿

This week, I mainly focused on the literature about how health states are constructed from vibration features or latent representations. One important finding is that “State Interpreter” is not a unified term in PHM. Related implementations appear under health indicator construction, degradation-stage detection, hidden-state discovery, change-point detection and RL state representation.

I summarized six representative implementation strategies.

The first strategy is a one-class health boundary. Healthy features are used to train an SVDD or One-Class SVM, and the distance to the healthy region becomes a continuous health indicator. For VibFM, the handcrafted features can be replaced by healthy `z_health` embeddings. This is the simplest baseline and requires only healthy-period data.

The second strategy comes from Juodelyte et al. Their implementation uses two AutoEncoders, concatenates the latent features, performs K-means with three clusters, orders the clusters by their average lifecycle position, and adds a fourth anomaly stage based on reconstruction error. For our project, the AutoEncoder latent can be replaced by time-ordered VibFM embeddings. This method provides interpretable degradation-stage probabilities, but the labels remain pseudo-labels rather than physical ground truth.

The third strategy combines change-point detection with a left-right HMM. It produces temporally consistent hidden degradation states and prevents unrealistic backward transitions. This method requires a continuous run-to-failure sequence, but it is more interpretable than immediately using a GRU.

The fourth strategy learns a continuous health indicator and an early degradation point. A small head on top of frozen VibFM could output both a health score and the probability that degradation has started. However, the original methods often train the encoder jointly with trend constraints, so this would be an adaptation rather than an exact reproduction.

The fifth strategy is a hybrid state that combines the vibration representation with temperature, load, speed, previous actions or maintenance history. This becomes important when these variables influence degradation dynamics or reward.

The sixth strategy comes from RL state-representation research, especially DeepMDP and bisimulation. A state should not only classify health; it should also preserve reward and action-conditioned transition information. Therefore, a later State Interpreter can include reward-prediction and next-state-prediction heads. This is what makes the state decision-ready.

Based on these papers, I propose a staged implementation. First, I would compare a one-class health indicator, K-means degradation stages, and a simple MLP producing both HI and stage probabilities. Second, I would add change-point detection or a left-right HMM. Only if history significantly improves reward or transition prediction would I introduce GRU or attention. Finally, the candidate states must be evaluated inside the RL maintenance environment.

As supporting implementation work, I created an independent State Interpreter repository and defined the model interface, common data contracts and adapters for VibFM, datasets and the Gearbox environment. Six software tests pass, but I have not yet trained the model or extracted real VibFM embeddings.

The next dependencies are the pretrained VibFM checkpoint, the exact preprocessing configuration, and the choice of the first run-to-failure dataset.

## 13. 建议最后询问老师

1. 您希望第一版 State Interpreter 优先输出连续 HI、退化阶段概率，还是两者同时输出？
2. 第一轮实验是否可以按 `One-Class HI → K-means stages → HMM` 的顺序推进？
3. 第一份数据是否优先使用公开 run-to-failure 数据？
4. 能否提供 VibFM checkpoint、对应代码版本和预处理参数？
5. Gearbox/Maintenance Arena 预计在什么时候用于 reward/transition 和 RL 闭环评价？
