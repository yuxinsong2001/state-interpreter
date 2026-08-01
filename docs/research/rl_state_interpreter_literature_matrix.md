# 第二轮文献矩阵：从 VibFM `z_health` 到 RL-usable State

检索日期：2026-07-25  
研究前提：固定使用 VibFM，不比较 encoder 优劣  
研究对象：位于 `z_health` 与 RL agent 之间的 State Interpreter

## 1. 检索问题

本轮检索回答五个问题：

1. 什么性质使一个 representation 成为可供 RL 使用的 state？
2. 如何检验 state 是否接近 Markov、能否保留 reward 和 transition 信息？
3. 单窗口 observation 不充分时，如何从历史构造 belief/history state？
4. 如何表达传感器噪声、隐藏退化和预测不确定性？
5. 如何证明 State Interpreter 真正改善维护决策，而不仅是分类或可视化效果？

## 2. 检索词

- `"state representation learning" control Markov reward relevant`
- `"DeepMDP" reward prediction transition representation`
- `"bisimulation" task relevant representation reinforcement learning`
- `"self-predictive representation" history POMDP`
- `"belief representation" partially observable deep RL`
- `"uncertainty aware" reinforcement learning maintenance`
- `"reinforcement learning" condition-based maintenance continuous state`
- `"Transformer" reinforcement learning prescriptive maintenance`

## 3. 纳入与排除标准

### 纳入

- 明确定义 RL state representation、state abstraction 或 belief state；
- 用 reward、transition、history 或 uncertainty 约束/评价 representation；
- 或者把健康/RUL信息接入 RL 维护决策并评价成本或 reward；
- 有正式论文页面、PMLR/OpenReview/期刊页面或可核验预印本。

### 排除

- 只做故障分类且没有 RL/state 理论；
- 只比较 encoder 准确率；
- 只展示 t-SNE 而没有 state sufficiency 或决策评价；
- 无法核验题目和来源的二手总结。

## 4. 核心文献矩阵

| 文献 | 理论问题 | State/方法 | 评价重点 | 代码 | 对 VibFM State Interpreter 的意义 | 优先级 |
|---|---|---|---|---|---|---|
| Lesort et al., 2018, *State Representation Learning for Control: An Overview* | 什么是适合控制的 representation？ | 将高维 observation 映射为低维、随时间变化且受 action 影响的 state；综述多类 SRL objective | 控制性能、低维性、可解释性及通用 state priors | 综述，无统一实现 | 用作理论总纲：`z_health` 只有在保留决策所需信息时才能称为 RL state | P0 |
| Jonschkowski & Brock, 2015, *Learning State Representations with Robotic Priors* | 如何用物理交互先验学习 state？ | temporal coherence、proportionality、repeatability、causality 等 priors | state 是否符合物理变化和 action-induced dynamics | 未核验到官方统一代码 | 可把 PHM 先验转成约束：健康状态应平滑，退化变化与载荷/action 有关，相同行为应产生相似变化 | P1 |
| Gelada et al., 2019, *DeepMDP* | representation 何时保留 MDP 决策信息？ | 同时预测 immediate reward 与下一 latent-state distribution | reward loss、transition loss、value/policy quality；给出表示质量的理论保证 | 未核验到作者维护的独立代码 | 最直接的理论模板：在 frozen `z_health` 后训练小型 state head，使其能预测 reward 和下一状态 | P0 |
| Zhang et al., 2021, *Learning Invariant Representations for RL without Reconstruction / Deep Bisimulation for Control* | 如何丢弃 task-irrelevant nuisance 而保留控制相关因素？ | 让 latent distance 对齐 bisimulation distance：相似 reward + 相似 next-state distribution 的 observation 应靠近 | distractor robustness、control return、generalization | [项目页含代码](https://sites.google.com/view/deepbisim4control) | 可用于抑制 `z_health` 中残留的转速、机器身份等 nuisance；比 reconstruction 更贴近 maintenance task | P0/P1 |
| Schwarzer et al., 2021, *Data-Efficient RL with Self-Predictive Representations* | 如何通过未来预测提高 sample efficiency？ | 根据当前 latent 和 action 预测未来多步 latent，并用数据增强保持一致性 | Atari 100k 的 sample efficiency、return、消融 | [作者代码](https://github.com/mila-iqia/spr) | 提供 `history/z_t + action → future state` 的辅助目标；但其视觉增强不能直接搬到振动数据 | P1 |
| Ni et al., 2024, *Bridging State and History Representations: Understanding Self-Predictive RL* | MDP state 和 POMDP history representation 有何共同结构？ | 将多种 state/history abstraction 统一为 self-predictive abstraction；讨论 stop-gradient 等优化 | MDP、distractor MDP、POMDP sparse reward | 未核验代码 | 给出是否需要 GRU/TCN 的理论条件：若单个 `z_t` 不充分，就用 history encoder 形成 self-predictive state | P0 |
| Wang et al., 2023, *Learning Belief Representations for Partially Observable Deep RL* | noisy/incomplete observation 下如何构造 reward-relevant belief？ | 解耦 belief model 与 policy；训练时利用真实 state，部署时从 history 推断 compact reward-relevant belief | 长期记忆、信息获取、state inference、PPO policy performance | [作者代码](https://github.com/awwang10/sphinx) | 对 PHM 很关键：真实 degradation 隐藏、振动只是 observation；可用模拟器 `statei/damage` 在训练期监督 belief head | P0 |
| Wei et al., 2023, *Set-membership Belief State-based RL for POMDPs* | 如何在有界传感器噪声下表达状态不确定性？ | belief 不只给点估计，而给包含真实 state 的集合边界 | 状态包含保证与下游 RL 表现 | 未核验作者代码 | 提醒不能只输出 stage argmax；可输出区间、集合或置信范围，让 policy 知道当前 state uncertainty | P1/P2 |
| Le Lan et al., 2022, *On the Generalization of Representations in RL* | representation 的拟合能力与泛化能力如何权衡？ | 用 effective dimension 等分析 state representation 的 approximation/generalization tension | 未见 state 的 value generalization bounds | 未核验代码 | State 过高维可能过拟合，过低维可能丢信息；应比较不同 state dimension 的跨 bearing/RL 泛化 | P1 |
| Zhao et al., 2023/2024, *TranDRL* | 如何把 prognostics 输出用于 maintenance RL？ | Transformer 从传感器预测 RUL；DRL 使用 RUL 等信息优化维护动作 | RUL误差、maintenance action、downtime/cost | 未核验作者代码；当前主要为 arXiv | PHM-RL 的模块化案例，但传给 RL 的主要是 RUL summary，不是完整 learned latent；适合作为 decision-level 对照 | P1 |
| Pan et al., 2021, *RL with Gaussian Processes for Condition-Based Maintenance* | 连续退化状态和不确定 dynamics 如何进入维护决策？ | continuous-state MDP；GP 近似 transition/value；优化长期平均成本 | long-run average cost；与离散 MDP 比较；battery case | 未核验作者代码 | 提供维护闭环评价范式：最终指标应是长期维护成本，而不只是 representation metric | P1 |
| Jha et al., 2020, *A Reinforcement Learning Approach to Health Aware Control Strategy* | 如何把 RUL/health 与 action-dependent degradation 联系起来？ | 用系统 transition data 和在线 RUL prediction 学习 health-aware control policy | 仿真控制性能与寿命目标 | 未核验作者代码 | 强调 action 会改变未来健康；State Interpreter 应尽量保留 action-conditioned degradation information | P1 |

## 5. 文献综合：什么才是 RL-usable State

本轮文献把“状态好不好”从静态分类问题提升为动态决策问题。

候选 state `s_t` 应尽可能满足：

### 5.1 Reward sufficiency

如果两个 observation 会导致显著不同的维护成本或 reward，它们不应被 State Interpreter 压缩成同一个 state。

可测试：

`s_t, a_t → r_t`

并比较加入完整历史后，reward prediction 是否显著改善。

### 5.2 Transition sufficiency

state 应包含预测下一状态分布所需的信息：

`s_t, a_t → p(s_{t+1})`

若使用单个 `z_health,t` 无法预测下一状态，而加入历史后明显改善，则当前 state 不是充分状态，需要 history encoder。

### 5.3 Policy/value sufficiency

被压缩到相近位置的 observation 应具有相近的最优 action 或 value。否则即使它们在健康程度上相似，对决策也不等价。

### 5.4 Nuisance invariance

如果转速、传感器或机器身份的变化不会改变 reward 和 transition，它们应尽量从 state 中被消除；如果工况会改变退化速度和 action 后果，则它们不能简单删除，而应进入 state。

### 5.5 Uncertainty awareness

振动只是对隐藏 degradation state 的有噪观测。State Interpreter 不应只给点估计：

`state = Stage 2`

更稳健的输出可能是：

`P(stage) + HI interval + out-of-distribution score`

### 5.6 History sufficiency

如果相同 `z_health,t` 可能分别对应“稳定健康”和“快速退化途中”，单窗口 embedding 会出现 observation aliasing。此时需要：

`s_t = g(z_{t-k:t}, a_{t-k:t-1}, maintenance history)`

## 6. 对当前架构的直接推导

### 6.1 不应直接把 `z_health` 当作最终 state

第一版应把它作为 observation embedding：

`e_t = z_health,t`

再由 State Interpreter 构造：

`s_t = g(e_t, history, operating conditions)`

### 6.2 第一候选 State Interpreter

```text
输入：
z_health,t
最近 k 个 z_health
工况 c_t
上一动作 a_(t-1)
距上次维护时间

输出：
health score h_t
stage probabilities p_t
trend Δh_t
uncertainty u_t
compact decision state s_t
```

早期不必一次实现全部输出。推荐递增：

1. `s_t = projection(z_t)`；
2. `s_t = [HI, stage probabilities]`；
3. 加入 trend 和 history；
4. 加入 uncertainty；
5. 加入 reward/transition auxiliary loss。

### 6.3 最重要的训练目标

基于 DeepMDP、bisimulation 和 self-predictive 文献，State Interpreter 可以采用：

`L = λ_r L_reward + λ_T L_transition + λ_H L_health + λ_U L_uncertainty`

其中：

- `L_reward`：预测 immediate/short-horizon maintenance reward；
- `L_transition`：预测 action-conditioned next state；
- `L_health`：HI、stage、damage 或其他 PHM 弱监督；
- `L_uncertainty`：概率校准或区间覆盖。

第一阶段冻结 VibFM，只训练 `g` 和辅助 heads，避免把多个问题混在一起。

## 7. 如何检验 Markov sufficiency

严格证明通常不可行，但可以做操作性诊断。

### Test A：历史增益测试

比较：

```text
模型 A：s_t, a_t → r_t, s_(t+1)
模型 B：s_t, history, a_t → r_t, s_(t+1)
```

若模型 B 稳定显著更好，说明 `s_t` 尚未包含足够历史信息。

### Test B：状态混叠检查

寻找 embedding 很近、但：

- reward 不同；
- next-state distribution 不同；
- 最优维护动作不同

的样本对。这类样本说明当前 state space 把决策上不同的状态错误合并。

### Test C：动作条件预测

比较是否输入 action：

`s_t → s_(t+1)`  
`s_t, a_t → s_(t+1)`

若 action 显著改善预测，说明维护动作/运行控制必须进入 dynamics model。

### Test D：不同历史长度

比较 `k = 1, 4, 8, 16...`，观察 transition/reward prediction 与 RL return 是否饱和。选择最短但足够的历史，避免无依据地使用长 GRU。

## 8. 更新后的实验矩阵

所有实验固定：

- 相同 VibFM checkpoint；
- 相同 `z_health`；
- 相同数据 split；
- 相同 RL algorithm 和主要超参数；
- 多个 random seeds。

| Interpreter | 输入 | 约束 | 目的 |
|---|---|---|---|
| Identity | `z_t` | 无 | 检查 VibFM embedding 直接交给 RL 的结果 |
| Compact projection | `z_t` | dimension bottleneck | 检查压缩是否提升 sample efficiency |
| PHM state | `z_t` | HI/stage supervision | 检查健康解释信息是否足够 |
| Predictive state | `z_t` 或 history | reward + transition prediction | 检查决策充分性 |
| Belief state | history | probabilistic/ensemble output | 处理 partial observability 和 uncertainty |
| Temporal state | history | HMM/GRU/TCN | 补充趋势和长期记忆 |

## 9. 双层评价协议

### 9.1 Representation-level

- reward prediction error；
- next-state/transition prediction error；
- 历史增益；
- state aliasing rate；
- health/damage/RUL probe；
- stage Macro F1；
- temporal consistency；
- uncertainty calibration/coverage；
- nuisance probe 与跨工况泛化；
- state dimension。

### 9.2 Decision-level

- cumulative/average return；
- maintenance cost；
- failure count/downtime；
- unnecessary maintenance；
- policy convergence speed；
- sample efficiency；
- policy stability across seeds；
- unseen bearing/工况上的 performance；
- noise、missing observation 与 domain shift 下的 robustness。

## 10. 第二轮检索对项目主线的修正

### 保留

- HI、degradation stage、change point 和 HMM；
- frozen VibFM 的模块化路线；
- 先简单 baseline，再复杂时序模型；
- 按 bearing/run/time 组织数据。

### 新增

- reward prediction；
- action-conditioned transition prediction；
- Markov/history sufficiency test；
- belief state 和 uncertainty；
- state aliasing 分析；
- RL closed-loop evaluation。

### 降级

- t-SNE/PCA 聚类图只能作为探索性证据；
- 单纯 Healthy/Damaged classification 不能证明 RL usefulness；
- 完整复现 Juodelyte AE 不是主线前置条件；
- “GRU + Attention”不再按模型名称决定，而由历史充分性测试决定。

## 11. 推荐阅读顺序

1. Lesort et al. (2018)：建立 SRL 总体概念；
2. DeepMDP (2019)：理解 reward + transition sufficiency；
3. Deep Bisimulation for Control (2021)：理解 task relevance 与 nuisance invariance；
4. Ni et al. (2024)：理解 state/history self-predictive abstraction；
5. Wang et al. (2023)：理解 belief representation 与 partial observability；
6. TranDRL、Pan et al.、Jha et al.：理解 PHM state 如何进入维护决策和成本评价。

## 12. 主要来源

- Lesort, T. et al. (2018). *State Representation Learning for Control: An Overview*. [arXiv](https://arxiv.org/abs/1802.04181)
- Jonschkowski, R., & Brock, O. (2015). *Learning State Representations with Robotic Priors*. [DOI](https://doi.org/10.1007/s10514-015-9459-7)
- Gelada, C. et al. (2019). *DeepMDP: Learning Continuous Latent Space Models for Representation Learning*. [PMLR](https://proceedings.mlr.press/v97/gelada19a.html)
- Zhang, A. et al. (2021). *Learning Invariant Representations for Reinforcement Learning without Reconstruction*. [项目页](https://sites.google.com/view/deepbisim4control)
- Schwarzer, M. et al. (2021). *Data-Efficient Reinforcement Learning with Self-Predictive Representations*. [arXiv](https://arxiv.org/abs/2007.05929)
- Ni, T. et al. (2024). *Bridging State and History Representations: Understanding Self-Predictive RL*. [ICLR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/666c1861d709bd84e20b6e0e02a2c223-Abstract-Conference.html)
- Wang, A. et al. (2023). *Learning Belief Representations for Partially Observable Deep RL*. [PMLR](https://proceedings.mlr.press/v202/wang23p.html)
- Wei, W. et al. (2023). *Set-membership Belief State-based Reinforcement Learning for POMDPs*. [PMLR](https://proceedings.mlr.press/v202/wei23d.html)
- Le Lan, C. et al. (2022). *On the Generalization of Representations in Reinforcement Learning*. [PMLR](https://proceedings.mlr.press/v151/le-lan22a.html)
- Zhao, Y. et al. (2023). *TranDRL: A Transformer-Driven Deep Reinforcement Learning Enabled Prescriptive Maintenance Framework*. [arXiv](https://arxiv.org/abs/2309.16935)
- Pan et al. (2021). *Reinforcement Learning with Gaussian Processes for Condition-Based Maintenance*. [DOI](https://doi.org/10.1016/j.cie.2021.107321)
- Jha, M. S. et al. (2020). *A Reinforcement Learning Approach to Health Aware Control Strategy*. [arXiv](https://arxiv.org/abs/2010.09269)

## 13. 当前证据局限

- RL representation 核心方法主要在视觉控制、机器人或通用 POMDP 上验证，迁移到高频振动 PHM 需要实验；
- TranDRL 当前核验来源为 arXiv，证据等级低于同行评审期刊；
- belief-state 方法常在训练时使用真实环境 state；当前 Gearbox simulator 可能提供该条件，真实 bearing 数据通常不提供；
- reward 和 action 只有在明确的 RL 环境中才存在，因此公开 run-to-failure 数据主要用于 representation-level 预训练和验证，不能独立完成最终闭环评价；
- 本轮是定向核心文献检索，不是 PRISMA 系统综述。

