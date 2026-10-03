# 可解除的候选事件解释器 v2：合成原型

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + synthetic implementation check
- Origin Date: 2026-09-19
- Verification Status: SYNTHETIC TESTED / NOT PREREGISTERED FOR REAL DATA / NO PHYSICAL ONSET VALIDATION
- Version Label: episodic_weak_event_v2_prototype

## 问题与范围

[v1开发结果](xjtu_condition3_weak_event_development_result_2026-09-19.md)和[只读分歧诊断](xjtu_condition3_weak_event_disagreement_diagnosis_2026-09-19.md)显示：B3_1 RMS在20–29步短暂连续超阈后，首次报警永久锁存，掩盖2335–2537步的晚期持续变化。v2只针对**事件生命周期和状态表达**建立独立原型；不修改v1的配置、代码、结果或结论，也不声称解决B3_2峭度稀疏超阈和物理标签缺失。

## 原型规则

单个设备/特征流使用前15次观测拟合固定总体均值`μ`和标准差`σ`；若`σ<1e-8`，标为`INVALID_REFERENCE`，不报警。之后仍用`μ+2σ`作同一个操作性阈值：连续5次严格超阈才进入`ACTIVE`，连续5次不超阈才退出回`MONITORING`；退出后可再次进入，episode ID累加。选择对称的5/5计数仅为最小原型定义，**不是从B3_1–B3_3结果优化出的参数，也未证明最优**。RMS和峭度需分别运行并保留不同provenance。

每步保留`deviation_z=(q−μ)/σ`、最近至多10个观测的因果斜率`trend`、确认后的活动持续步数`duration_since_confirm`、活动episode ID、进入/退出标志。校准期间不输出事件且拒绝步号缺口；**监测阶段**步号缺口输出`UNKNOWN_GAP`、中断活动事件但**不宣称设备恢复**，之后重新监测。候选开始步只能在确认后显示，不回填既有输出。`ACTIVE`表示算法确认的相对振动变化，不是故障或健康阶段真值。

实现文件：`src/state_interpreter/episodic_weak_event.py`。合成测试：`tests/test_episodic_weak_event.py`，覆盖短波动不进入、进入→退出→再次进入、监测缺口不当作恢复、校准缺口拒绝、因果前缀不变性、连续强度/趋势、无效参考、跨bearing隔离和无效输入原子拒绝。合成测试7 passed；全量测试221 passed。未读取或重放XJTU-SY任何真实bearing，B3_4/B3_5继续保护。

## 下一验证门槛

1. 优先取得独立的故障起始、人工检查、维修或振动异常记录；若无，研究目标只能是“持续变化事件”，不能评价真实onset准确率。
2. 在不改v1的前提下，为v2另建输入/输出契约和预注册。指标应包括episode次数、活动占比、解除/再确认、早期瞬态与末段持续变化能否分离，以及跨bearing的一致性；三个run不能当成大量独立样本。
3. v2仅修复“首次报警锁存”这一机制问题；同一阈值的进出仍可能抖动，B3_2峭度缺乏五次持续超阈也不会自动变成有事件。若要改变特征、阈值或滞回幅度，必须另立版本并寻找独立评价数据，不能在B3_4/B3_5上反复调参。
