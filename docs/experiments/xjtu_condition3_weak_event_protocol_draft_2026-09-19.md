# Condition 3 因果候选事件弱标签协议（草案）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-19
- Verification Status: DESIGN ONLY / NOT PREREGISTERED / NOT EXECUTED
- Version Label: condition3_weak_event_protocol_draft_v1
- 输入依据：[健康锚点来源审计](xjtu_condition3_health_anchor_inventory_2026-09-19.md)、[局部特征核查](xjtu_condition3_local_feature_audit_2026-09-19.md)、[XJTU-SY工程基线审计](../research/xjtu_sy_feature_lstm_code_audit_2026-09-18.md)。
- 方法来源：第三方工程基线的[统计onset detector](https://github.com/thfmn/xjtu-sy-bearing/blob/7d7231c582961741bde629da6731e6c169d88785/src/onset/detectors.py)借鉴了早期参考、`mean+2σ`与连续5次超阈值；本草案把“前20%完整寿命”改为固定前15次，**不宣称复现上游方法或真实故障标签**。

## 研究问题与目标

在不使用未来测量和完整寿命的前提下，能否从当前振动特征产生可追溯、可解释的**候选状态变化事件**？本阶段只检验事件算法的可运行性、稳定性和信息边界，不评价“真实故障起点准确率”，也不训练新的Encoder、GRU或分类器。

实验单位为bearing run，不把连续测量点当独立设备。开发范围仅`Bearing3_1`–`Bearing3_3`已冻结的65维特征缓存。`Bearing3_4`验证与`Bearing3_5`盲测继续不读；不读取第三方对它们的逐bearing onset数值。

## 弱标签契约

| 字段 | 定义 | 允许的解释 |
| --- | --- | --- |
| `phase` | `CALIBRATING`（前15次）、`MONITORING`、`ALARMED` | 算法运行阶段，不是物理健康阶段 |
| `event_flag_t` | 初始为0，确认报警时变为1并保持1 | 是否已经出现本算法确认的事件；0是“未报警”，**不是已证实健康** |
| `event_confirm_step` | 满足连续5次超阈值时的**当前**步号 | 在线可获得的确认时间，不追溯回填到第1次超阈值 |
| `candidate_start_step` | 五次连续超阈值的首步，可事后附录记录 | 分析用候选起点；不能当作当时已可知的在线输出 |
| `event_strength_t` | 当前参考标准化超阈幅度，有限值 | 相对振动变化量，非物理损伤百分比 |
| `provenance` | `rms_threshold_v1`或`kurtosis_threshold_v1` | 标签生成来源和版本，便于区别真实注释 |

若全程无确认报警，`event_confirm_step=null`，不把最后一次测量强制设为候选onset。设备终止是另一个run-level事实，不反推前面的弱标签。

## 最小算法设计（执行前仍须冻结）

1. 按`step_id`升序处理每条bearing；前15次仅拟合并冻结参考统计量，`phase=CALIBRATING`。第16次才允许开始监测。不使用`sequence_length`、终点、未来样本或完整寿命比例来设置阈值。
2. 主基线`rms_threshold_v1`：每步`q_t=(h_rms+v_rms)/2`。参考为前15个`q`的均值`μ`和总体标准差`σ`；阈值`μ+2σ`。从第16次起，连续5个`q_t>阈值`，在第5个超阈点确认事件并锁存`ALARMED`。不回填在线输出。
3. 预列对照`kurtosis_threshold_v1`：使用水平/垂直峭度均值，采取相同的15步、`μ+2σ`和连续5次规则。单独报告，不在看到开发结果后择优或按bearing切换。
4. 若任一参考`μ/σ/阈值`非有限，或`σ<1e-8`，该通道标记`INVALID_REFERENCE`且不给事件；不得临时加ε或换特征掩盖问题。若测量缺失或步号不连续，重置连续计数并记录缺口；不得把不连续记录拼成5次连续观测。
5. `event_strength_t=(q_t-μ)/max(σ,1e-8)`仅在参考有效且监测状态下报告，另存原始`q_t`和阈值。该强度可能为负；它只是标准化偏离，不是损伤量。具体字段名及数据类型在实现前固定。

`mean+2σ`、连续5次来源于开源工程思路，固定15步来自当前项目的在线校准约束。参数不是依据本轮B3_1–B3_3报警效果优化，也没有证据证明它们最优；这是**待核查的操作性定义**。与官方“最大振幅超过正常阶段最大值10倍”的**试验停止规则不同**，不能混称。

## 只读开发评价（没有准确率）

- 完整记录每个分支在三条开发bearing的参考有效性、是否报警、确认步号、完整run中报警比例、距记录终点的测量次数，以及无报警情况。`event_fraction=confirm_step/(N-1)`和`lead_to_recorded_end=N-1-confirm_step`仅供**事后描述**，不得进入在线推理或调阈值。
- 对照RMS与峭度两个独立预列规则的报警一致性和时间差，但它们来自同一振动系统，不是独立真值。不能把两者一致率称为诊断准确率。
- 检查确定性、单向锁存、因果前缀不变性（添加未来数据不能改变既有时刻输出）、不同bearing之间必须显式`reset()`、文件缺失处理、非有限值拒绝。
- 不利用B3_1–B3_3的已看过曲线选择更好规则、修改阈值或定义“成功的报警时间段”。即使三条都报警，也只证明这个规则能触发，不证明抓到了真实损伤起点。

## 与State Interpreter的关系

后续可以把`event_flag`作为**带provenance的辅助上下文**，与`Level/Trend/Movement`并列输出；不得把它直接训练成“真实健康阶段”标签。若以后尝试用伪标签训练模型，必须把伪标签生成器视为teacher，评价不能再用同一teacher输出算准确率，否则会形成循环验证。必须寻求专家记录、独立检查或独立数据集作为外部评价。

## 进入实现前的门槛

1. 先确认教师/数据提供方是否有独立的故障开始、人工检查、阈值触发记录；若有，修改评价目标并单独建立新协议。
2. 固定输入缓存哈希、确切代码/配置、输出路径与模式；合成序列测试边界条件、前缀不变性和保护集拒绝。
3. 首轮仅运行B3_1–B3_3开发数据，明确结果为`weak_event_feasibility`而非真实onset验证。**本草案尚未冻结或执行**，B3_4/B3_5不开。

## 风险与完整性

主要风险是早期15次并非真实健康、均值与标准差受异常点影响、峭度方向不统一、终点由另一振幅规则决定、以及在已经看过开发轨迹后作事后方法选择。记录所有三条bearing和两个分支，不挑好例；弱标签只用于机制演示。11/11统计谬误已在设计层检查：重点防伪重复、由算法一致性推断准确率、事后挑阈值、把run-level终点外推到逐步状态及相关当因果；本草案无p值或显著性主张。
