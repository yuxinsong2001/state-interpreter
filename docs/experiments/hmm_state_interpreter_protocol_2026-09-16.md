# HMM State Interpreter探索实验协议

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：plan + run
- Origin Date：2026-09-16
- Verification Status：UNVERIFIED（运行前）
- Version Label：hmm_interpreter_protocol_v1

## 研究问题

在固定AutoEncoder embedding的前提下，具有显式状态转移约束的HMM能否比当前距离基线更稳定地表达退化阶段，并减少短期随机波动造成的状态摆动？

老师认为Bearing1_4的特殊表现可能来自随机波动。因此，本实验不再以“修复Bearing1_4”为目标，也不基于该bearing单独调参，而是把随机波动作为所有bearing都可能具有的序列特征。

## 固定条件

- 输入使用已生成的`z=8` latent CSV，不重新训练AutoEncoder。
- 距离基线保持`calibration_steps=15`、`temporal_window=10`。
- HMM固定为三个状态，使用对角高斯发射分布。
- HMM只允许自循环或向下一个阶段转移。
- HMM仅在原训练bearing（Bearing1_1–Bearing1_3）上拟合。
- 推理使用因果过滤，只依赖当前和过去的embedding。
- 两种方法从相同时间步开始评分，避免不同预热区间造成不公平比较。

## HMM输出

- `stage`：当前最大后验概率对应的离散阶段；
- `stage_probabilities`：三个阶段的过滤概率；
- `expected_stage`：将阶段概率组合为`[0,1]`连续值；
- `confidence`：最大阶段概率。

状态编号表示模型学习到的时间阶段，不能在实验前直接命名为“健康、退化、损坏”。只有检查出现位置、持续时间和转移行为后，才能讨论其健康含义。

## 评价方法

- 与normalized lifetime的Spearman相关性：仅表示时间排序，不代表物理损伤精度；
- 相邻状态变化的标准差与平均绝对变化：描述短期波动；
- backward-step比例：描述连续状态向较早阶段回摆的频率；
- 离散阶段切换次数与平均置信度：检查HMM是否频繁跳变或过度确定。

## 证据边界

五个bearing及其历史结果均已被观察，因此本实验属于探索性比较，不是盲测。训练bearing上的结果仅用于描述模型行为；Bearing1_4和Bearing1_5也不能恢复为未见测试集。本实验可以判断实现是否可用以及是否值得进入更严格的新数据协议，但不能单独证明跨设备泛化。

## 后续门槛

只有当HMM实现通过单元测试、概率归一化正确、因果前缀测试一致，并且没有明显退化为固定时间分段后，才进入GRU实验。GRU和Attention应继续使用相同embedding和公共评分区间。

首轮结果保存为逐时间步CSV、bearing指标CSV、HMM参数和JSON报告。绘图不是执行依赖，后续可从逐时间步CSV独立生成，避免因本地缺少可选绘图库而阻断核心实验。

## 执行状态（2026-09-16）

实验已成功执行，状态为exploratory / analyzed。第一版HMM在Bearing1_4上出现恒定状态输出，并在其他bearing上表现出接近1的过度置信，未达到进入复杂模型前所设定的质量门槛。完整结果见`hmm_state_interpreter_result_2026-09-16.md`。
