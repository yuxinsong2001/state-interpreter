# HMM log-domain过滤验证

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：run + validate
- Origin Date：2026-09-16
- Verification Status：VERIFIED
- Version Label：hmm_log_filter_validation_v1

## 验证问题

第一版HMM接近1的置信度和Bearing1_4恒定状态，是否由概率域计算`exp(log emission)`时的数值下溢造成？

## 修改与控制变量

仅将在线过滤改为log-domain：使用log-sum-exp计算预测概率和归一化后验。AutoEncoder、latent CSV、训练bearing、EM拟合、状态数、发射方差、转移约束、距离基线和公共评分时间点全部保持不变。新结果写入独立目录，不覆盖v1。

新增极端embedding测试，确认即使普通概率密度会下溢到0，log-domain过滤仍能返回有限且归一化的状态概率。全量测试结果由74项增加为75项，全部通过。

## 精确比较

| 比较项 | 结果 |
|---|---:|
| 逐时间步行数 | 616 |
| 离散阶段不一致数 | 0 |
| 三个阶段概率最大绝对差 | 0 |
| 连续期望阶段最大绝对差 | 0 |
| 置信度最大绝对差 | 0 |
| HMM参数文件SHA-256 | 完全一致 |

两个版本的Bearing级Spearman相关性、阶段切换次数和backward-step比例也完全一致。结果文件的文本长度不同，是浮点序列化格式和报告元数据不同造成的，不代表数值差异。

## 结论

数值下溢假设被排除。当前数据上的过度置信和Bearing1_4状态塌缩来自已学习的窄对角高斯发射分布与左到右阶段约束，而不是旧过滤公式的浮点错误。log-domain实现仍应作为正式实现保留，因为它对极端输入更安全。

这不说明所有HMM都不适用，只说明当前三状态、对角高斯、无额外正则化的HMM没有优于距离基线。HMM v1可以锁定为对照，下一阶段可以进入GRU方案设计。

## Fallacy Scan（11/11）

| 检查项 | 状态 | 说明 |
|---|---|---|
| Simpson's paradox | 未发现 | 比较在每个时间步与bearing内完成 |
| Ecological fallacy | 未发现 | 未从聚合值推断单个设备 |
| Berkson's paradox | 注意 | 仍限于同一个已见数据集 |
| Collider bias | 不适用 | 未引入控制变量 |
| Base-rate neglect | 不适用 | 不属于故障分类评估 |
| Regression to the mean | 未发现 | 没有选择极端对象后比较改善 |
| Survivorship bias | 无法确认 | 数据集删失信息未改变也未新增 |
| Look-elsewhere effect | 未发现新问题 | 本次只验证一个预先明确的数值假设 |
| Garden of forking paths | 注意 | 整体HMM研究仍属于已见数据上的探索 |
| Correlation ≠ causation | 未作因果主张 | 只验证两种计算实现是否数值一致 |
| Reverse causality | 不适用 | 本次验证不涉及方向性因果推断 |

## 工件

- 新配置：`configs/xjtu_hmm_interpreter_v1_1.json`
- 新结果：`results/2026-09-16_hmm_interpreter_log_filter_check/`
- 正式实现：`src/state_interpreter/hmm_temporal.py`
- 极端数值测试：`tests/test_hmm_temporal_interpreter.py`
