# 组会汇报：State Interpreter for VibFM

汇报日期：2026-07-27  
本周工作范围：理论定位、文献路线、Gearbox 数据审计、第一版软件架构与接口验证

## 一、建议开场

本周我首先重新明确了课题边界。当前研究不再比较不同 encoder，而是固定使用 VibFM，并研究如何把 VibFM 输出的健康表征 `z_health` 转换成紧凑、可解释且适合强化学习决策的状态。

因此当前主线是：

```text
vibration
    ↓
STFT + frozen VibFM
    ↓
z_health
    ↓
State Interpreter
    ↓
interpretable and decision-relevant state
    ↓
RL maintenance decision
```

## 二、对黑板问题的回答

### 1. VibFM 在系统中的位置

VibFM 负责从振动信号提取 latent representation。它是固定的上游 feature extractor，而不是本课题要重新比较或替换的模型。

当前更准确的关系是：

```text
raw vibration → STFT → VibFM → z_health → State Interpreter → RL state
```

`z_health` 是 State Interpreter 的输入，但不应自动等同于最终 state。

### 2. State Interpreter 的核心任务

State Interpreter 需要回答：

- 当前设备处于什么健康状态？
- 是否进入退化阶段？
- 退化程度和趋势如何？
- 状态判断的不确定性有多大？
- 当前信息是否足以支持维护决策？

它不是单纯的 Healthy/Damaged 分类器。它应把高维 embedding 转换为适合 RL 使用的状态：

```text
s_t = [health indicator,
       stage probabilities,
       trend,
       uncertainty,
       relevant operating conditions]
```

具体分量仍需通过实验确定。

### 3. 如何理解黑板上的 embedding space

老师在黑板上画出的 Healthy 和 Damage 聚类表示：首先要检查 `z_health` 是否形成与健康状态有关的结构。

可使用 PCA、UMAP 或 t-SNE 做可视化，但二维聚类图只能用于探索，不能单独证明 state 已经适合 RL。

还需要检验：

- embedding 是否随退化形成连续轨迹；
- 是否仍受到转速、负载、传感器或机器身份影响；
- 相近 embedding 是否具有相近 reward 和 transition；
- 是否存在 embedding 相似，但最优维护动作不同的 state aliasing。

### 4. 什么时候使用 GRU + Attention

GRU + Attention 是候选方法，不是预先确定的方案。

第一版先采用单窗口 MLP：

```text
z_health,t → MLP → HI + stage probabilities → compact state
```

之后进行 history-sufficiency test：

```text
只使用 s_t, a_t
        vs.
使用 s_t + history, a_t
```

比较二者预测 reward、next state 和 RL performance 的能力。如果加入历史后显著改善，才说明单个 `z_health,t` 不足，需要 GRU、TCN、HMM 或 Attention。

### 5. State 什么时候才算 decision-ready

一个 representation 能区分 Healthy 和 Damaged，并不代表它已经适合决策。Decision-ready state 至少应满足：

- 紧凑；
- 与 reward 和维护动作相关；
- 尽可能接近 Markov state；
- 保持合理的时间一致性；
- 对噪声和无关工况稳健；
- 能表达必要的不确定性；
- 能在不同设备或工况间迁移。

最终必须进行两层评价：

1. Representation level：健康相关性、时间一致性、预测性、稳健性和可解释性；
2. Decision level：RL 收敛、sample efficiency、maintenance cost、failure/downtime 和 policy stability。

## 三、文献调研得到的研究路线

老师提供的综述与任务材料说明，State Interpreter 的关键不是只生成退化标签，而是构造 RL-usable state。

当前整理出的四类方案为：

1. 单窗口 Linear/MLP：`z_health → HI + stage`；
2. 时序 Interpreter：`z_health history → TCN/GRU/HMM → health + trend`；
3. Hybrid Interpreter：融合工况、previous action 和 maintenance history；
4. Multi-task Interpreter：联合预测 HI、stage、uncertainty、reward 和 next state。

推荐顺序是由简单到复杂逐步验证，不直接从 GRU + Attention 开始。

Juodelyte et al. (2022) 对本课题的作用主要是提供 latent clustering、退化阶段生成、阶段重叠和逆向转移评价的参考。它不是完整的 RL State Interpreter，因为没有解决 reward relevance、Markov sufficiency 和 RL decision evaluation。

## 四、本周具体完成的工作

### 1. 审计 Gearbox 项目

确认 Gearbox 环境生成的是连续 episode：

```text
action
→ torque/load
→ degradation
→ vibration
→ observation
→ reward
```

环境中可以获得：

- vibration；
- GearIn/GearOut damage；
- torque/load；
- action/reward；
- episode、step 和 `nolc`；
- next observation 和 done。

但目前这些字段没有统一导出成标准 transition；Bearing1–4 的 degradation 仍为 `None`。现有代码中名为 `interpreter` 的函数实际负责 reward 计算，也不等于本课题的 State Interpreter。

因此，Gearbox 当前适合做接口和 RL 管线测试，但不适合作为主要科学证据。

### 2. 建立独立 State Interpreter 仓库

为了避免模型与 VibFM 或 Gearbox 的内部代码耦合，建立了独立仓库：

```text
state-interpreter/
├── src/state_interpreter/
├── tests/
├── README.md
├── pyproject.toml
└── .gitignore
```

未来 VibFM、Gearbox 和真实数据集通过 adapter 接入，而不是复制核心代码。

### 3. 建立第一版模型骨架

第一版采用两层 MLP：

```text
z_health
    ↓
MLP
    ├── health indicator
    └── four-stage probabilities
               ↓
compact RL state
```

当前接口测试假设：

```text
输入 fake z_health：[8, 256]
输出 HI：[8, 1]
输出 stage probabilities：[8, 4]
输出 compact state：[8, 5]
```

这里使用 fake embedding 只为验证软件接口和 shape，不代表模型已经学会健康状态。

### 4. 建立统一数据契约

定义了三个标准对象：

```text
RawMeasurement
    = episode/time + vibration + conditions + optional labels

EmbeddingSample
    = RawMeasurement + z_health

StateTransition
    = current + action + reward + next + done
```

这样可以保证真实数据、VibFM、Gearbox 和 RL 使用同一种时间与字段语义。

### 5. 建立 Adapter 接口

- `VibFMAdapter`：`vibration → z_health`；
- `DatasetAdapter`：按设备/run 和时间读取真实数据；
- `GearboxAdapter`：从环境采集标准 transition。

目前这些是接口定义，还不是 checkpoint 推理或具体数据集实现。

### 6. 验证结果

- State Interpreter 模型与数据契约共 6 项测试通过；
- 可以检测错误 embedding shape；
- 可以阻止跨 episode transition；
- 可以阻止非递增时间顺序；
- Gearbox 两步 smoke test 通过。

## 五、当前结论边界

本周已经完成：

- 研究问题收敛；
- 理论要求与评价路线整理；
- Gearbox 数据能力审计；
- 独立软件架构；
- 第一版 MLP 与数据接口；
- fake embedding 和数据一致性测试。

本周尚未完成：

- 真实 VibFM checkpoint 推理；
- 真实 `z_health` 提取；
- 正式 HI/stage 标签定义；
- MLP 训练；
- 性能实验；
- RL 闭环评价。

因此不能汇报分类准确率、健康趋势或 RL 收益。目前所有数值结果均为软件测试结果，不是研究实验结果。

## 六、数据与实验策略

主要实验应优先使用公开实测 run-to-failure 数据，例如 FEMTO-ST/PRONOSTIA、NASA IMS 或 XJTU-SY。

需要注意：这些数据参与过 VibFM 预训练，可用于检查 `z_health` 是否支持状态解释，但不能单独证明对完全未见数据的泛化。后续还需要未参与预训练的数据或老师提供的真实项目数据。

Gearbox 仿真的定位是：

- 管线 smoke test；
- adapter 和 transition 接口验证；
- RL 闭环测试；
- 在物理模型验证后作为受控补充实验。

## 七、下一步计划

### 获得 VibFM checkpoint 后

1. 确认对应 GitHub 版本或 commit；
2. 确认 STFT、normalization 和输入 shape；
3. 确认 `z_health` 的真实维数与读取位置；
4. 实现真实 `VibFMAdapter`；
5. 按 unit/run/time 批量提取 embedding。

### 数据侧

1. 确定第一份 run-to-failure 数据集；
2. 实现具体 `DatasetAdapter`；
3. 保留严格时间顺序并按 bearing/run 划分数据；
4. 明确 HI 和 degradation-stage 标签来源；
5. 先检查 embedding trajectory，再训练单窗口 MLP。

### 模型侧

1. 建立 Identity 与 MLP baseline；
2. 评价 HI、stage、时间一致性和 nuisance robustness；
3. 通过 history-gain test 决定是否加入 GRU/TCN；
4. 最后加入 reward/transition 预测并接入 RL。

## 八、建议向老师确认的问题

1. Could you please provide the pretrained VibFM checkpoint and indicate which repository version or commit it belongs to?
2. Which dataset should I prioritize for the first State Interpreter experiment: a public run-to-failure dataset or the current gearbox project?
3. For the first version, should the state primarily represent a continuous health indicator, degradation-stage probabilities, or both?
4. Should the first evaluation focus on representation quality before integration into the RL maintenance environment?
5. Is the current Gearbox simulator intended only for pipeline/RL integration, or should it also be treated as a formal source of training data?

## 九、3 分钟口头汇报稿

This week, I first refined the scope of the State Interpreter task. I now treat VibFM as a fixed upstream encoder, so the main research question is no longer which encoder is better, but how to transform the VibFM health embedding into an interpretable and decision-relevant state for reinforcement learning.

Based on the review material, I found that separating healthy and damaged samples is not sufficient. An RL-usable state should also preserve reward-relevant and action-conditioned transition information, be approximately Markovian, temporally consistent, robust to nuisance factors, and interpretable.

Therefore, I organized the candidate methods into four levels: a single-window MLP, a temporal model such as GRU or TCN, a hybrid model including operating conditions and action history, and a multi-task model with health, stage, uncertainty, reward and transition prediction. I would start with the MLP baseline and only introduce GRU or attention if history significantly improves reward or next-state prediction.

I also audited the Gearbox environment. It produces continuous episodes and provides vibration, gear damage, torque, action, reward and next observations internally. However, these fields are not yet exported as one standardized transition, and bearing degradation is currently not implemented. Therefore, I would use the simulator for pipeline and RL integration tests, but not as the main scientific evidence.

On the implementation side, I created an independent State Interpreter repository. I implemented the first MLP interface, a unified data contract, and adapter interfaces for VibFM, measured datasets and the Gearbox environment. Six software tests pass. However, these are only interface tests using fake embeddings. I have not yet trained the model or produced real experimental results.

The next step is to obtain the VibFM checkpoint and preprocessing configuration, select the first run-to-failure dataset, extract real time-ordered health embeddings, and then train and evaluate the simple MLP baseline before considering temporal models.

## 十、30 秒总结

本周最重要的结果不是训练了一个模型，而是把问题定义和实验路线固定下来：

```text
fixed VibFM
→ inspect z_health
→ simple interpretable baseline
→ test history sufficiency
→ add temporal/hybrid information only when justified
→ evaluate both representation and RL decision quality
```

当前的软件接口已经准备好。下一阶段的关键依赖是 VibFM checkpoint、预处理配置和第一份正式数据集。
