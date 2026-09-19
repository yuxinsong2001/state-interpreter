# XJTU-SY Condition 3 健康／事件锚点来源审计

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate / source and label audit
- Origin Date: 2026-09-19
- Verification Status: ANALYZED（来源核对；未产生新标签或训练）
- Version Label: condition3_health_anchor_inventory_v1
- 官方来源：[数据集作者仓库](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets)及随数据包提供的`Introduction_to_XJTU-SY_Bearing_Dataset.pdf`，第2–3页。
- 工程来源：[特征缓存生成代码](../../src/state_interpreter/feature_materialization.py)、[本次训练入口](../../scripts/train_source_conditional_mmd_condition3_development.py)、[缓存manifest](../../cache/xjtu_condition3_feature_lstm_v1/manifest.json)。
- 第三方方法参照：[XJTU-SY工程基线代码审计](../research/xjtu_sy_feature_lstm_code_audit_2026-09-18.md)及其[公开仓库](https://github.com/thfmn/xjtu-sy-bearing)。
- 本次未读取B3_4/B3_5信号、缓存、模型结果或逐bearing衍生标签；没有新训练。

## 审计问题

在当前XJTU-SY Condition 3实验里，能否取得独立于测量序号、足以监督连续设备状态或真实退化起点的标签？“标签存在”还必须区分**原始提供**、**可由试验终止规则推得**、**算法估计**和**人工事后整理**，不能混为ground truth。

## 可用信息与证据等级

| 信息 | 来源与粒度 | 证据性质 | 能否直接监督逐时刻State？ |
| --- | --- | --- | --- |
| 每次振动测量 | 原始CSV，水平/垂直通道；每分钟记录1.28秒 | 原始观测 | 只能作为输入，不能自带健康标签 |
| 测量序号／总文件数 | 原始目录及官方表2；B3_1=2538、B3_2=2496、B3_3=371次 | 观测顺序＋完整运行长度 | 可构造归一化时间/RUL代理；未来失效时刻在在线运行时未知，不等于损伤 |
| 终止事件 | 官方说明：试验持续到某通道最大振幅超过正常阶段最大振幅的10倍 | 作者报告的试验停止规则，run-level事件 | 可界定记录的终点；正常阶段最大值的确切计算区间和逐条触发记录未随当前缓存给出，不能凭此标注每个中间时刻 |
| 最终失效部件 | 官方表2：B3_1外圈；B3_2内圈＋滚动体＋保持架＋外圈；B3_3内圈 | 运行结束后的故障部件描述 | 仅run-level，不可复制给每个时间点当阶段标签；不同故障模式可能影响特征迁移 |
| 真实逐分钟损伤量／人工确认onset时刻 | 官方说明与本地当前缓存未发现 | **当前不可用**；不排除数据发布者另有未取得的记录 | 不能作为现成监督信号 |
| 第三方curated onset | `thfmn/xjtu-sy-bearing`的统计检测＋逐bearing整理 | 派生伪标签，不是数据集作者的逐时刻物理真值 | 可在新协议中作为弱监督/候选事件；禁止用保护bearing标签选规则或评判严格盲测 |

当前缓存manifest明确只有`features`、`step_ids`、`source_paths`。实际训练目标由`selected / (sequence_length - 1)`构造，因而是**线性测量时间比例**。这解释了为什么前一步看到的B3_3“前中段下降、末端突升”会与线性目标存在张力，但不能因此断言真实损伤路径就是该曲线。官方终止条件定义的是试验结束，不是退化起点或连续健康等级。

## 对下一阶段的限制与可行路径

1. **当前不具备真实逐时刻监督标签。** 不应把`normalized_lifetime`重命名为`damage`，也不应把B3_3最后20%自动标为“故障阶段”。
2. **最可靠的现有锚点是run-level终点和最终故障部件。** 可用于明确终止事件、做故障模式分层描述，但不足以直接训练逐时刻阶段分类。
3. **若研究目标仍是在线State Interpreter，下一轮可考虑弱监督、因果事件检测或分阶段建模；但必须先定义“事件”的操作标准。** 候选例如固定前15次健康参考＋只使用过去观测的振幅/峭度变化检测。该事件是算法触发，不可称物理损伤起点；规则、阈值、比较分支、评价集均须在查看保护集前锁定。第三方按“前20%完整寿命”校准的方法不满足在线条件。
4. **需要教师/数据提供方补充信息时，最有价值的问题**是：是否有每条bearing试验的故障起始记录、人工检查时间点、停止阈值的实际计算记录、以及终点故障部件的检查报告？若没有，研究问题应明确降级为“基于弱标签的相对状态和事件提示”，而非真实健康阶段识别。
5. **当前先不启动新训练。** 先把标签定义与评估指标写成独立预注册，避免再次仅更换模型而沿用可能不合适的线性时间目标。B3_4/B3_5继续不读。

## 完整性与风险

- 官方表2和停止规则已与随数据包PDF第3页视觉核对；作者GitHub README确认这是15条run-to-failure数据。官方说明没有列逐分钟损伤/起点标签；“未在已审计材料发现”不等于证明发布者绝无其他元数据。
- 第三方onset生成方法是另一个工程项目，不代表原始数据集标签。其具体逐bearing值没有在本次读取。
- 11/11统计谬误核查：此文是来源清单，无p值/效应检验；重点防止由run-level故障部件推断逐点标签（生态推断）、只看完整失效run的选择偏差、把时间相关当损伤因果、把派生onset冒充真值、以及事后选择触发阈值；其余类型在本次来源审计中不直接适用或无法凭文档排除。
