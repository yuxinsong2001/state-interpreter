# 2026-07-27 组会汇报终稿

## 题目

**From VibFM Embeddings to RL-usable States: Literature Review and a Progressive Experimental Strategy**

## 汇报的核心观点

文献中不同的 State Interpreter 方法不应被理解为四个必须一次性全部实现的竞争模型，而应理解为逐步增加状态能力的实验路线：

```text
先建立最简单的状态
        ↓
检查它缺少什么
        ↓
只在有证据时增加时间、工况或决策约束
        ↓
选择满足任务要求的最简单模型
```

---

# 第一部分：研究问题

## 1. 老师黑板上的核心问题

当前系统可以表示为：

```text
vibration
    ↓
STFT
    ↓
frozen VibFM
    ↓
z_health embedding
    ↓
State Interpreter
    ↓
RL state
    ↓
maintenance decision
```

我的研究问题不是重新选择 encoder，而是：

> Given a fixed VibFM encoder, how should `z_health` be transformed into an interpretable and decision-relevant state?

这里有两个需要验证的问题：

1. `z_health` 是否已经包含健康和退化结构？
2. 即使包含健康信息，它是否已经足够支持 RL 决策？

因此：

```text
z_health ≠ automatically decision-ready state
```

## 2. 老师材料提供的评价标准

老师提供的两份材料主要帮助定义“最终 state 应满足什么要求”：

- compact；
- interpretable；
- temporally consistent；
- robust；
- approximately Markov；
- uncertainty-aware；
- reward/action relevant；
- transferable。

Google Scholar 文献调研则用于回答：

> 已有研究通过哪些方法实现这些能力？

---

# 第二部分：文献中的四类实现思想

## 3. 方法一：健康空间距离

代表方向：

- Health Indicator construction；
- SVDD；
- One-Class SVM；
- healthy-reference distance。

代表文献：

Zhou et al. (2016), *Bearing Performance Degradation Assessment Using Lifting Wavelet Packet Symbolic Entropy and SVDD*。

核心思想：

> 使用健康期样本定义正常区域，以样本偏离健康区域的程度表示连续退化程度。

迁移到 VibFM 后：

```text
healthy z_health
→ healthy embedding region
→ distance from healthy region
→ continuous HI
```

它回答：

> 当前状态离健康状态有多远？

主要能力：

- 连续健康程度；
- 异常偏离；
- 低标签需求。

主要缺口：

- 不一定时间平滑或单调；
- 不直接提供退化阶段；
- 不一定保留决策信息。

## 4. 方法二：聚类与退化阶段

代表文献：

Juodelyte et al. (2022), *Predicting Bearings' Degradation Stages for Predictive Maintenance in the Pharmaceutical Industry*。

核心思想：

> 在 latent space 中发现不同数据区域，再结合生命周期中的时间位置为 cluster 赋予退化语义。

迁移到 VibFM 后：

```text
time-ordered z_health
→ clustering
→ clusters ordered by lifecycle position
→ degradation-stage pseudo-labels
→ stage probabilities
```

它回答：

> 当前样本属于哪个退化阶段？

主要能力：

- 离散、可解释阶段；
- 无需逐窗口人工标签；
- 可以分析 embedding space。

主要缺口：

- cluster 是 pseudo-label，不是物理真值；
- 需要连续 run-to-failure 序列；
- 可能出现不合理的阶段逆向跳转；
- “越晚越严重”是一项时间先验。

## 5. 方法三：时间状态模型

代表方向：

- Change-Point Detection；
- left-right HMM；
- GRU/TCN；
- Attention。

代表文献：

- Cartella and Sahli：Continuous Hidden Markov Model；
- Wang et al.：topological representation + HMM；
- Singleton et al.：hidden health-state discovery。

核心思想：

> 设备健康状态是随时间演化的隐藏过程，当前窗口可能不足以表达趋势。

迁移到 VibFM 后：

```text
z_health,t-k:t
→ temporal state model
→ health state + trend
```

它回答：

> 当前状态是如何演化到这里的？

主要能力：

- 趋势；
- 变点；
- 时间一致性；
- 隐状态转移。

主要缺口：

- 必须有连续序列；
- 模型更复杂；
- 可能在少量设备上过拟合；
- Attention 本身不能保证可解释或决策相关。

## 6. 方法四：决策相关状态

代表文献：

- Lesort et al.：State Representation Learning for Control；
- DeepMDP；
- Deep Bisimulation for Control；
- belief/self-predictive representation。

核心思想：

> 能区分 Healthy 和 Damaged，并不代表 state 已经适合 RL。State 还应保留 reward 和 action-conditioned transition 信息。

评价关系：

```text
s_t, a_t → reward_t
s_t, a_t → s_(t+1)
```

候选实现：

```text
z_health/history
→ State Interpreter
├── HI
├── stage probabilities
├── reward prediction
└── next-state prediction
        ↓
compact decision state
```

它回答：

> 当前状态对未来退化和维护动作意味着什么？

主要能力：

- reward relevance；
- action-conditioned dynamics；
- 检测 state aliasing；
- 直接面向 RL。

主要缺口：

- 需要 action、reward 和 transition；
- 公开 run-to-failure 数据通常没有维护动作；
- 需要 RL 环境；
- 训练和评价成本最高。

---

# 第三部分：这四种方法是否都要实现？

## 7. 它们不是同一层级的四个竞争模型

前两种方法主要定义状态的健康语义：

```text
连续语义：HI
离散语义：degradation stage
```

后两种方法主要检查状态是否充分：

```text
时间充分性：是否需要 history？
决策充分性：是否保留 reward/action/transition？
```

因此不能简单做：

```text
四种模型全部实现
→ 比较分类准确率
→ 选择最高者
```

因为它们解决的问题不同。

## 8. 推荐原则：逐步增加能力

推荐实验逻辑：

```text
简单 state
→ 表示层诊断
→ 时间充分性诊断
→ 决策充分性诊断
→ 必要时增加复杂度
```

最终原则：

> Select the simplest State Interpreter that is sufficient for the downstream maintenance decision.

也就是：

> 选择满足状态充分性要求的最简单模型。

## 9. 第一阶段：必须建立的低成本基线

### Baseline 0：Identity

```text
state = z_health
```

目的：

> 检查 State Interpreter 是否真的比直接使用 VibFM embedding 更有价值。

### Baseline 1：Continuous HI

```text
z_health
→ healthy-space distance
→ HI
```

目的：

> 检查 `z_health` 是否包含连续退化方向。

第一轮可以只使用一种 One-Class 方法，不需要同时实现 One-Class SVM、SVDD 和所有距离方法。

### Baseline 2：Degradation stages

```text
time-ordered z_health
→ clustering
→ time-ordered stages
```

目的：

> 检查 embedding space 是否形成可解释阶段。

### Baseline 3：Single-window MLP

```text
z_health
→ MLP
→ HI + stage probabilities
```

目的：

> 检查单个 embedding 是否足以同时表达连续和离散健康语义。

第一阶段建议完成这四个低成本 baseline，因为它们构成后续所有复杂模型的必要对照。

## 10. 第二阶段：如何决定是否使用 HMM、GRU 或 Attention

首先检查第一阶段的状态：

- HI 是否随时间剧烈波动？
- 阶段是否频繁出现逆向跳转？
- 是否无法稳定检测退化起点？
- 单个 embedding 是否混淆“稳定健康”和“快速退化中”的状态？

然后进行 history-gain test：

```text
模型 A：
s_t, a_t → reward_t / next state

模型 B：
s_t + history, a_t → reward_t / next state
```

如果加入历史没有显著增益：

```text
保留简单单窗口模型
```

如果加入历史有显著增益：

```text
simple smoothing / CPD
→ left-right HMM
→ GRU or TCN
→ Attention only if it adds further value
```

因此老师黑板上的 GRU + Attention 应被理解为候选方向，而不是项目开始时必须采用的结构。

## 11. 第三阶段：如何决定是否需要 Decision-relevant Interpreter

公开 run-to-failure 数据可以评价：

- HI；
- stage；
- trend；
- change point；
- temporal consistency。

但它通常不能完整评价：

- maintenance action；
- reward；
- action-conditioned transition；
- policy performance。

当 Gearbox/Maintenance Arena 提供 RL transition 后，再比较：

```text
PHM state:
HI + stage

Predictive state:
HI + stage
+ reward prediction
+ next-state prediction
```

如果 predictive state 能够：

- 更好预测 reward；
- 更好预测 action 后的状态变化；
- 提高 RL sample efficiency；
- 降低 maintenance cost；
- 减少 failure/downtime；

才保留这些额外模块。

---

# 第四部分：整体实验决策树

## 12. 实验路线

```text
获取真实 z_health
        ↓
Identity + HI + stages + single-window MLP
        ↓
表示是否具有健康语义？
        │
   否 ──┴── 是
   │        ↓
检查 VibFM/标签   检查时间一致性
和数据定义         ↓
             状态是否稳定？
                │
          ┌─────┴─────┐
          │           │
         是           否
          │           ↓
          │      CPD/left-right HMM
          │           ↓
          │      历史是否仍有增益？
          │        ┌──┴──┐
          │       否     是
          │        │      ↓
          │        │   GRU/TCN
          │        │      ↓
          │        │ Attention 是否有额外增益？
          └────────┴─────────────→ RL environment
                                      ↓
                            reward/transition sufficiency
                                      ↓
                            是否改善维护决策？
```

## 13. 每一步的停止条件

复杂度不是无限增加的。

- 如果 One-Class HI 已经稳定支持决策，不需要复杂时序模型；
- 如果 HMM 已经解决时间一致性，不一定需要 GRU；
- 如果 GRU 与 TCN 没有显著差异，选择更简单稳定的模型；
- 如果 Attention 没有带来额外收益，不保留 Attention；
- 如果 reward/transition auxiliary loss 没有改善 RL，不保留这些 heads；
- 如果工况不影响 reward/dynamics，应把它当作 nuisance，而不是加入 state。

---

# 第五部分：如何评价和选择最终模型

## 14. Representation-level

评价：

- HI 与 damage、RUL 或 lifetime position 的关系；
- monotonicity；
- trendability；
- robustness；
- stage Macro F1；
- 状态逆向跳转次数；
- EDP/change-point detection；
- 跨 bearing 和跨工况泛化；
- uncertainty calibration；
- state dimension。

这些指标回答：

> 这个 state 是否具有合理的健康语义？

## 15. Decision-level

评价：

- reward prediction；
- transition prediction；
- history gain；
- state aliasing；
- RL convergence；
- sample efficiency；
- cumulative return；
- maintenance cost；
- unnecessary maintenance；
- failure/downtime；
- policy stability。

这些指标回答：

> 这个 state 是否真正帮助维护决策？

不能只用分类准确率选择最终 State Interpreter。

---

# 第六部分：本周完成的实验准备

## 16. 独立仓库初始化

为了保持模块边界清晰，State Interpreter 采用独立仓库：

```text
VibFM
  ↓ z_health
State Interpreter repository
  ↓ compact state
Gearbox / RL environment
```

当前仓库包含：

- 第一版单窗口 MLP 接口；
- `RawMeasurement`；
- `EmbeddingSample`；
- `StateTransition`；
- `VibFMAdapter`；
- `DatasetAdapter`；
- `GearboxAdapter`；
- 模型和数据契约测试。

当前结果：

- 6 项软件测试通过；
- fake `z_health [8,256]` 能得到 `[8,5]` compact state；
- 能检测错误 shape、跨 episode transition 和错误时间顺序。

结论边界：

> 这些是软件接口测试，不是真实 State Interpreter 实验。

目前尚未：

- 获取真实 checkpoint；
- 提取真实 `z_health`；
- 正式定义 HI/stage 标签；
- 训练模型；
- 接入 RL；
- 产生性能结果。

---

# 第七部分：下一步

## 17. 数据与 VibFM

1. 获得 pretrained VibFM checkpoint；
2. 确认代码版本、STFT 和 normalization；
3. 确认 `z_health` 的实际输出与维度；
4. 选择第一份 run-to-failure 数据；
5. 按 bearing/run/time 组织数据；
6. 提取真实、按时间排序的 embedding。

## 18. 第一轮实验

固定相同的：

- VibFM checkpoint；
- embedding；
- 数据划分；
- bearing-level split；
- evaluation protocol。

比较：

1. Identity；
2. One-Class HI；
3. K-means stages；
4. single-window MLP。

然后根据时间一致性和 history gain 决定是否进入 HMM、GRU/TCN 或 Attention。

## 19. 需要老师确认

1. 第一版 state 是否优先同时输出 HI 和 stage probabilities？
2. 是否认可按照 `simple baselines → temporal sufficiency → decision sufficiency` 的路线推进？
3. 第一份数据应使用公开 run-to-failure 数据还是当前 Gearbox 项目？
4. 能否提供 VibFM checkpoint、对应代码版本和预处理配置？
5. Maintenance Arena 何时可以用于 action、reward 和 transition 评价？

---

# 5 分钟口头汇报稿

This week, I focused on how the different State Interpreter ideas from the literature should be selected and evaluated.

The main conclusion is that the methods should not be treated as four models that all have to be fully implemented and compared only by classification accuracy. They solve different levels of the problem and should form a progressive experimental strategy.

The first two groups mainly define the health semantics of a state. Distance-based methods define a continuous health indicator by measuring how far a VibFM embedding is from the healthy region. Clustering-based methods, such as the approach of Juodelyte et al., discover degradation stages in the latent space and assign temporal meaning to the clusters.

The third group adds temporal sufficiency. Change-point detection, left-right HMMs, GRUs or TCNs use embedding history to model trends and hidden state transitions. However, I would not start directly with GRU and attention. I would first test whether a single-window state is unstable and whether adding history significantly improves reward or next-state prediction.

The fourth group addresses decision sufficiency. Based on DeepMDP and bisimulation, an RL state should preserve reward and action-conditioned transition information. This part can only be evaluated when an interactive maintenance environment provides actions, rewards and transitions.

Therefore, I propose to start with four low-cost baselines: the raw VibFM embedding as an identity baseline, a one-class health indicator, clustering-based degradation stages, and a single-window MLP that outputs both HI and stage probabilities.

If these states are temporally stable and history provides no additional information, I would keep the simple model. If they show unstable transitions or history significantly improves prediction, I would first add change-point detection or a left-right HMM, and only then compare GRU or TCN. Attention would only be retained if it provides measurable additional value.

Finally, when the RL environment becomes available, I would compare a normal PHM state with a predictive state trained using reward and transition objectives. The final model should be the simplest State Interpreter that is sufficient for the maintenance decision.

As preparation, I initialized an independent State Interpreter repository and defined the MLP interface, standard measurement and transition contracts, and adapters for VibFM, measured datasets and the Gearbox environment. Six software tests pass, but no real VibFM embeddings have been extracted and no model has been trained yet.

The next steps are to obtain the VibFM checkpoint and preprocessing configuration, select the first run-to-failure dataset, and run the simple baseline experiments before deciding whether temporal or predictive models are necessary.
