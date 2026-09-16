# 第一版HMM State Interpreter探索结果

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：run + validate
- Origin Date：2026-09-16
- Verification Status：ANALYZED
- Version Label：hmm_interpreter_result_v1

## 实验目的

在不重新训练AutoEncoder的条件下，比较当前距离基线与三状态左到右高斯HMM，检查HMM能否利用显式状态转移减少随机波动，并保持有意义的退化时间排序。

HMM仅在Bearing1_1–Bearing1_3上拟合。推理采用因果过滤，只读取当前和过去的embedding。两种方法都从step 15开始评分。所有bearing过去均已被观察，因此本次结果属于探索性分析，不是盲测。

## 结果

| Bearing | Split | 距离基线ρ | HMM ρ | HMM阶段切换 | HMM平均置信度 |
|---|---|---:|---:|---:|---:|
| Bearing1_1 | train | 0.964 | 0.854 | 1 | 1.000000 |
| Bearing1_2 | train | 0.994 | 0.834 | 2 | 1.000000 |
| Bearing1_3 | train | 0.954 | 0.914 | 2 | 0.999997 |
| Bearing1_4 | validation | 0.849 | 无法计算 | 0 | 1.000000 |
| Bearing1_5 | old holdout | 0.950 | 0.816 | 1 | 0.999986 |

HMM将backward-step比例大幅压低，但不能直接将其解释为更稳健。原因是它同时产生了接近1的置信度和很少的阶段切换。Bearing1_4在公共评分区间中保持同一个阶段，连续期望阶段为常数，所以Spearman相关性无法计算。这属于状态塌缩，而不是成功消除了随机波动。

## 结论

第一版HMM的软件实现是可运行的：概率归一化、左到右转移和因果前缀一致性均通过测试。但当前三状态对角高斯设置没有优于距离基线，并且过度确定。较小的短期变化只是模型离散化和状态饱和带来的结果，不能据此宣称HMM更好。

这个结果支持保留距离Level作为基线，同时把HMM v1作为后续GRU和Attention比较中的一个失败但有信息量的候选。若继续研究HMM，应优先检查发射方差正则化、状态数、概率校准和跨bearing分布偏移，而不是根据Bearing1_4单独调参。

## 统计与解释边界

- 没有对五个bearing执行显著性检验：样本量很小，且训练bearing与开发bearing角色不同。
- Spearman相关性只评价时间排序，不评价物理损伤、RUL或维护决策价值。
- 训练bearing结果不是泛化证据；Bearing1_4与Bearing1_5也已被观察。
- HMM的阶段编号是时间状态，不自动等于健康、退化和损坏。
- 本次结果不能用于声称HMM造成了性能改善。

## Fallacy Scan（11/11）

| 检查项 | 状态 | 说明 |
|---|---|---|
| Simpson's paradox | 未发现 | 逐bearing报告，没有用聚合值掩盖方向反转 |
| Ecological fallacy | 未发现 | 没有从工况均值推断单个bearing |
| Berkson's paradox | 注意 | 只使用一个已筛选公开数据集，外部泛化受限 |
| Collider bias | 不适用 | 未加入控制变量 |
| Base-rate neglect | 不适用 | 未报告故障分类灵敏度或阳性预测值 |
| Regression to the mean | 未发现 | 没有按极端得分选择对象后宣称改善 |
| Survivorship bias | 无法确认 | 当前数据审计未提供删失或未完成run的完整信息 |
| Look-elsewhere effect | 注意 | 属于多指标探索；未执行或选择性报告显著性检验 |
| Garden of forking paths | 注意 | 方法是在已见数据上探索，不能转为验证性结论 |
| Correlation ≠ causation | 注意 | 时间相关性不能证明健康损伤因果关系 |
| Reverse causality | 注意 | 时间顺序明确，但normalized lifetime不是损伤机制标签 |

## 工件

- 配置：`configs/xjtu_hmm_interpreter_v1.json`
- 实验协议：`docs/experiments/hmm_state_interpreter_protocol_2026-09-16.md`
- 结果：`results/2026-09-16_hmm_interpreter_exploratory/`
- 单元测试：`tests/test_hmm_temporal_interpreter.py`

第一次尝试因本地缺少可选的matplotlib而在读取数据前退出；移除非必要绘图依赖并获得用户明确确认后，第二次运行成功。生成的`experiment_report.json`仍包含一句关于展示用绘图归一化的旧提示，但实际输出清单没有图像；该提示不影响任何数值，脚本随后已删除这句过期提示。

## 后续数值验证

HMM在线过滤随后改为log-domain，并通过极端embedding测试。新旧版本在616个时间步上的阶段概率、期望阶段、置信度和离散阶段完全一致，最大绝对差为0。因此，过度置信和Bearing1_4状态塌缩不是概率域下溢伪影。完整验证见`hmm_log_filter_validation_2026-09-16.md`。
