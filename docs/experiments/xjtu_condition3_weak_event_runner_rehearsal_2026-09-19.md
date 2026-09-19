# Condition 3 弱事件执行入口：合成演练

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: implementation and synthetic validation
- Origin Date: 2026-09-19
- Verification Status: RUNNER READY / SYNTHETIC TESTED / REAL DEVELOPMENT NOT EXECUTED
- Frozen preflight config: `configs/xjtu_condition3_weak_event_development_v1.json`
- Separate execution lock: `configs/xjtu_condition3_weak_event_execution_lock_v1.json`

新增`src/state_interpreter/weak_event_development.py`和`scripts/run_weak_event_condition3_development.py`。执行入口仅接受`--check-only`或`--execute-development`；不接受任意数据路径或bearing参数。进入真实读取前先核对固定配置、三份开发缓存、detector、evaluator和runner的哈希，且拒绝已存在的结果或执行记录。冻结配置未被追加或改写；执行代码使用独立锁文件固定。

输出契约为`candidate_event_trajectories.csv`、`candidate_event_summary.csv`和`report.json`，另在`records/`保存执行状态及报告哈希。逐步轨迹分别保留两个provenance，不择优。汇总包含每条bearing/规则的参考有效性、报警确认步号、缺口数、相对于记录终点的事后测量次数；`event_fraction`以测量位置而非任意step ID计算，只用于事后描述。无报警保留空值，不强制把终点当事件。`physical_onset_accuracy=null`。

执行器在生成输出前验证所有输入数组：恰好65维、样本数匹配、严格递增的非负整数step ID、所有特征有限。缺失step允许但重置连续超阈计数；非有限特征或重复/倒序step会终止，不跳过坏行。RMS和峭度以独立实例运行，同一bearing内按时间顺序处理，不在bearing之间共享参考。

合成集成测试验证两个分支同时输出、一次报警不回填、缺口重置、保护bearing拒绝、非有限值拒绝、无效参考及代码哈希篡改拒绝。`python scripts/run_weak_event_condition3_development.py --check-only`通过：只哈希开发文件字节，**不解析NPZ**；全量测试**214 passed**。真实`--execute-development`尚未调用，B3_4/B3_5未读取。

下一门槛：明确批准完整命令`python scripts/run_weak_event_condition3_development.py --execute-development`后，才进行一次性B3_1–B3_3开发运行；随后只读汇总候选事件及失败模式。无独立物理故障锚点时不能评价真实onset准确率。
