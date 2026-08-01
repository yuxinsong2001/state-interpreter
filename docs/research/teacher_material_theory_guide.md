# 老师任务材料：State Interpreter 理论导向

整理日期：2026-07-25

## 1. 材料来源与证据边界

本笔记基于老师此前提供的两份 PDF：

1. `__Learned State Representations for RL in PHM_ A Literature Review__ Copy.pdf`
   - 31 页；
   - PDF 元数据作者为 Giuseppe Mannone；
   - 内容是 learned state representation、PHM 与 RL 的综述。
2. `State Interpreter für RL in PHM Copy.pdf`
   - 5 页；
   - 内容是德语研究任务说明；
   - PDF 元数据作者显示为 ChatGPT Canvas，因此其中内容应视为老师提供的任务材料，而不是同行评审论文。

两份材料用于确定研究目标、理论概念和评价维度。综述中的论文、DOI、性能数字和“尚无研究”等文献性断言仍需通过原始来源核验，不能直接视为已验证证据。

## 2. 老师最初定义的核心问题

任务书第 1 页的上位研究问题是：

> 如何开发和评价一个面向 PHM 传感器数据的 learned State Interpreter，使 RL agent 能够从中获得稳健、可解释且与维护相关的状态，用于自适应维护决策？

因此，State Interpreter 不是普通故障分类器。它是：

```text
PHM observations
        ↓
State Interpreter
        ↓
RL-usable state
        ↓
RL policy
        ↓
maintenance action
```

其最终价值必须由“是否支持更好的决策”判断，而不能只看分类准确率或聚类图。

## 3. 对 State 的理论定义

31 页综述第 8 页将 State Representation Learning 写成：

`φ(o_t) = s_t`

其中：

- `o_t`：时刻 `t` 的原始或高维 observation，例如振动、温度、工况和历史序列；
- `φ`：State Interpreter；
- `s_t`：提供给 RL agent 的紧凑状态。

当前使用 VibFM 后，可具体化为：

```text
o_t
 = 当前振动窗口 + 可用工况 + 必要历史

VibFM(o_t)
 = z_health,t

State Interpreter
 = g(z_health,t, history, operating conditions, uncertainty)

s_t
 = RL agent 使用的状态
```

这说明 `z_health` 不一定已经等于最终 RL state。它更准确的定位是 State Interpreter 的核心输入或中间表征。

## 4. 一个 RL-usable State 必须满足什么要求

综合任务书第 2-4 页与综述第 8-11 页，至少有七项要求。

### 4.1 紧凑性

状态维度不能大到让 RL agent 重新面对 state-space explosion，但也不能过度压缩而丢失故障模式、趋势或决策信息。

### 4.2 决策相关性

状态应包含与 reward、维护动作和未来风险有关的信息，而不只是能够重建输入信号。

### 4.3 近似 Markov 性

给定当前状态 `s_t` 和动作 `a_t`，应尽可能包含预测下一状态和回报所需的信息。若单个 `z_health,t` 无法表达退化速度或历史趋势，则 State Interpreter 需要加入：

- 相邻 embedding；
- trend/slope；
- elapsed time；
- 上一次维护信息；
- GRU、TCN、HMM 或其他记忆机制。

因此，时序模型的理论理由不是“模型更复杂”，而是补偿 partial observability，使状态更接近 Markov。

### 4.4 时间一致性

相邻时刻的状态变化应平滑、合理，并与退化动力学相符。对于不可逆退化，不应频繁出现：

`Critical → Healthy → Critical`

### 4.5 稳健性

状态应尽量不受噪声、传感器差异、转速、负载和机器身份等 nuisance factors 干扰。

### 4.6 可解释性

状态应能与工程上可理解的量建立联系，例如：

- health/degradation level；
- RMS 或频谱变化；
- damage；
- RUL/失效风险；
- operating condition；
- uncertainty。

### 4.7 可迁移性

State Interpreter 应尽可能在不同 bearing、机器、工况或 simulation-to-real 场景中保持有效。

## 5. State 不应只等于一个 degradation label

老师任务书第 2 页列出对 RL 决策可能重要的信息：

- degradation state；
- trend information；
- uncertainty；
- operating conditions；
- signs of impending failure。

因此，一个较完整的候选状态可以写成：

`s_t = [h_t, p_t, Δh_t, u_t, c_t]`

其中：

- `h_t`：连续 health indicator；
- `p_t`：离散 degradation-stage probabilities；
- `Δh_t`：趋势或退化速度；
- `u_t`：不确定性；
- `c_t`：工况信息。

这只是候选设计，不是已经确定的最终结构。它比单独输出 Healthy/Damaged 更接近老师对 RL state 的要求。

## 6. 两阶段评价框架

材料显示 State Interpreter 必须同时经过 representation-level 和 decision-level 评价。

### 6.1 第一层：状态表示本身是否合理

- 健康与损伤状态是否可分；
- 是否与 degradation、damage 或 RUL 相关；
- 是否保持时间一致性；
- 是否对噪声和工况变化稳健；
- 是否跨设备和模拟保持稳定；
- 是否能解释各状态分量；
- 是否保留预测未来变化所需的信息。

### 6.2 第二层：状态是否真正帮助 RL

- RL learning curve/convergence speed；
- sample efficiency；
- cumulative reward；
- maintenance cost；
- unplanned failures/downtime；
- policy stability；
- 跨工况 policy robustness；
- 相同 RL 算法在不同 State Interpreter 下的结果。

这纠正了当前文献矩阵偏重 PHM 状态识别的问题：HI、K-means 和 HMM 是中间评价，不是最终证明。最终必须把候选 state 放进 RL 环境。

## 7. 模块化与端到端

任务书第 2 页明确提出 modular 与 end-to-end 的比较问题。31 页综述第 9-10、13 页指出，模块化预训练通常更适合数据稀缺、噪声高和安全要求高的场景。

在当前 VibFM 前提下，第一阶段应采用模块化路线：

```text
冻结 VibFM
    ↓
训练 State Interpreter
    ↓
冻结或单独验证 State Interpreter
    ↓
接入现有 RL agent
```

优点：

- 能分别判断 representation、interpreter 和 policy 的问题；
- 减少 RL sparse reward 对 VibFM 的破坏；
- 训练成本更低；
- 更容易解释和复现实验。

端到端微调可作为后期扩展，但需要明确证明它相对于模块化方案的收益。

## 8. VibFM 确定后的研究边界

早期材料把 raw/STFT/wavelet 以及 CNN/TCN/AE 比较列为候选研究任务。但后续组会已确定 VibFM 是上游，因此当前不再把 encoder selection 作为主要研究变量。

应保留的理论内容：

- `φ(o_t)=s_t` 的 RL state representation 问题；
- 紧凑、决策相关、近似 Markov、稳健和可解释的要求；
- modular 与 end-to-end 的架构选择；
- representation-level 与 RL decision-level 的双层评价；
- 噪声、工况、数据稀缺、Sim-to-Real 和安全问题。

应更新的旧内容：

```text
早期问题：
哪种 raw/transformed representation 和 encoder 最适合？

当前问题：
给定 VibFM z_health，哪种 State Interpreter 最能产生 RL-usable state？
```

## 9. Juodelyte 论文在新理论框架中的位置

Juodelyte et al. (2022) 主要帮助解决：

- 如何从 latent representation 生成 degradation stages；
- 如何利用时间排序赋予 cluster 语义；
- 如何评价阶段重叠和逆向转移。

它没有充分解决：

- 状态是否 Markov；
- 是否包含 uncertainty 和 operating conditions；
- 是否与 reward 和 action 相关；
- 是否提高 RL sample efficiency 或 maintenance reward；
- 是否适合作为完整 RL state。

所以它只是 State Interpreter 的“退化阶段构造模块”参考，而不是完整理论方案。

## 10. 更新后的工作主线

### 阶段 A：理论要求

1. 定义 observation、State Interpreter、RL state 和 action 的边界；
2. 定义 RL-usable state 的必要性质；
3. 定义模块化架构和评价协议。

### 阶段 B：VibFM 表征检查

1. 获得 checkpoint；
2. 提取按 unit/run/time 排序的 `z_health`；
3. 检查健康相关性、nuisance leakage 和时间轨迹；
4. 判断单窗口 embedding 是否足够。

### 阶段 C：State Interpreter 候选

1. `z_health → HI`；
2. `z_health → degradation-stage probabilities`；
3. `z_health sequence → trend/change point/HMM`；
4. 增加 uncertainty 和 operating conditions；
5. 形成紧凑状态向量 `s_t`。

### 阶段 D：表示层评价

评价区分度、退化相关性、时间一致性、稳健性、可解释性、跨设备迁移和近似 Markov 性。

### 阶段 E：RL 闭环评价

将候选 `s_t` 接入同一个 RL 环境和算法，比较：

- 学习速度；
- cumulative reward；
- 维护成本；
- 故障和停机；
- policy stability；
- 跨工况泛化。

### 阶段 F：复杂模型与端到端扩展

只有简单 State Interpreter 无法满足 Markov 性、时间一致性或决策性能时，再加入 GRU、Attention、TCN 或端到端微调。

## 11. 当前最重要的研究问题

### 主问题

> 在固定使用 VibFM 的前提下，如何把 `z_health` 与必要的时间、工况和不确定性信息转换成紧凑、可解释、近似 Markov 且与维护决策相关的 RL state？

### 子问题

1. 单个 `z_health,t` 是否已包含当前健康与下一步退化所需的信息？
2. State Interpreter 应输出连续 HI、阶段概率，还是组合状态？
3. 需要多长历史才能使状态接近 Markov？
4. 如何减少 `z_health` 中残留的 operating-condition/nuisance 信息？
5. 如何估计和表达不确定性？
6. representation-level 指标能否预测 RL decision-level 表现？
7. 模块化 State Interpreter 是否足够，何时才需要联合微调？

## 12. 对现有文档的影响

- `literature_matrix.md` 中的 HI、degradation stage 和 HMM 文献仍然相关；
- 后续文献检索需要补充 RL State Representation、Markov sufficiency、reward relevance、uncertainty-aware state 与 representation-to-policy evaluation；
- `presentation_notes.md` 后续汇报应从“状态识别”扩展到“RL-usable state”；
- Juodelyte 复现不是主线终点，只是学习阶段构造的一个支线；
- 最终实验必须包含 RL 闭环评价，不能只停留在 embedding 可视化或分类指标。

## 13. 固定 VibFM 后的四种 State Interpreter 架构候选

老师原始材料中的 AutoEncoder、CNN、TCN、Hybrid 以及带 RUL/Anomaly/Degradation 信息的方案，最初覆盖了从原始信号编码到 RL state 的完整过程。现在 VibFM 已经固定为上游 encoder，因此可将仍然相关的思想重新组织为以下四种 State Interpreter。

这四种方案不是互斥的四选一，而是一条由简单到复杂的递进实验路线。

### 13.1 方案一：单窗口轻量 Interpreter

```text
z_health,t
    ↓
Linear / MLP
    ↓
HI + stage probabilities
    ↓
compact RL state
```

作用：

- 检查单个 VibFM embedding 是否已经足以描述当前健康状态；
- 建立参数少、易解释的最低复杂度 baseline；
- 避免在尚未证明历史必要时直接使用复杂时序模型。

候选输出：

- 连续 HI；
- Healthy/Degradation/Critical 的概率；
- 低维投影状态。

主要局限：

- 无法显式表示退化速度；
- 若相同 `z_health,t` 可能来自不同历史轨迹，会出现 state aliasing；
- 不天然满足 Markov sufficiency。

### 13.2 方案二：TCN/GRU 时序 Interpreter

```text
z_health,t-k ... z_health,t
            ↓
        TCN / GRU
            ↓
health + trend + compact state
```

作用：

- 从连续 embedding 中提取趋势、退化速度和长期依赖；
- 缓解单窗口 observation 的 partial observability；
- 使状态更加接近 Markov。

选择 TCN、GRU 或其他时序模型的依据应是 history sufficiency test，而不是预先假定复杂模型一定更好。需要比较不同历史长度，并检验加入历史后 reward/transition prediction 和 RL 表现是否改善。

主要局限：

- 需要可靠的 unit/run ID 和时间顺序；
- 独立测量数据不能直接使用；
- run-to-failure 设备数量少时容易过拟合；
- 状态解释比轻量模型困难。

### 13.3 方案三：Hybrid Interpreter

```text
z_health history ──────────┐
转速、负载、温度等工况 ─────┼→ Fusion → compact RL state
上一动作、距上次维护时间 ─────┘
```

作用：

- 区分“无关 nuisance”和“会改变退化动力学或决策后果的工况”；
- 将感知信息、运行条件和维护历史组合成决策状态；
- 为 action-conditioned transition prediction 提供必要输入。

关键原则：

- 如果某个变量只改变测量外观而不改变 reward/transition，应尽量实现 invariance；
- 如果变量会改变退化速度、故障风险或维护动作效果，就必须保留在 state 中。

主要局限：

- 需要数据集中存在同步工况和动作记录；
- 不同数据源的时间尺度、归一化和缺失值需要处理；
- fusion 模块可能重新引入机器身份或数据集泄漏。

### 13.4 方案四：Multi-task Interpreter

```text
                         ┌→ HI
z_health / history ──────├→ degradation stage
                         ├→ anomaly probability
                         ├→ RUL / failure risk
                         ├→ uncertainty
                         ├→ reward prediction
                         └→ next-state prediction
                                  ↓
                         compact RL state
```

作用：

- 用多个辅助任务约束 state，使其同时包含 PHM 语义和 RL 决策信息；
- 将老师材料中的 RUL、Anomaly、Degradation information 与第二轮文献中的 reward/transition sufficiency 结合；
- 提供对人可解释的输出，同时向 RL agent 提供紧凑 latent state。

候选总损失：

`L = λ_H L_health + λ_S L_stage + λ_RUL L_RUL + λ_r L_reward + λ_T L_transition + λ_U L_uncertainty`

所有任务不必同时启用。应通过消融实验判断每个辅助任务是否改善 representation-level 或 decision-level 指标。

主要局限：

- 不同损失可能互相冲突；
- 标签可得性不同；
- loss weights 增加调参负担；
- RUL 或 pseudo-label 质量差时可能误导 state；
- 需要防止通过未来信息构造标签造成数据泄漏。

## 14. 四种方案的递进关系

推荐顺序：

```text
方案一：单窗口 MLP
检查当前 z_health 是否已经足够
        ↓
方案二：TCN/GRU
仅在历史显著改善预测或决策时加入
        ↓
方案三：Hybrid
加入会影响 dynamics/reward 的工况和维护历史
        ↓
方案四：Multi-task
用 PHM 与 RL 辅助目标共同约束最终 state
```

它们也可以组合成最终候选：

```text
VibFM
  ↓
z_health sequence
  ↓
TCN/GRU temporal module
  ↓
operating-condition/action fusion
  ↓
multi-task heads
├── HI
├── stage
├── trend
├── uncertainty
├── reward prediction
└── next-state prediction
  ↓
compact RL state
```

实验中必须同时保留简单方案一，否则无法证明增加时序、融合和多任务模块的必要性。
