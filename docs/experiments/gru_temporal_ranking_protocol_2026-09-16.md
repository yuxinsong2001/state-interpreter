# GRU 时间排序目标：配对消融实验协议

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：run / pre-execution protocol
- Origin Date：2026-09-16
- Verification Status：UNVERIFIED（执行前协议）
- Version Label：gru_temporal_ranking_v1

## 问题和假设

上一轮固定 checkpoint 对照显示，把 hidden-distance 改成 predicted-z distance 没有明显改善 seed 稳定性。本轮检验显式约束健康读出的训练目标是否有帮助，而不改变 GRU 架构。

排序假设：同一 bearing 的较晚测量总体上比早期测量更偏离初始状态。该假设允许通过软惩罚被违反，但可能不适合健康平台期、工况变化或非单调退化。这里使用时间顺序作为弱监督；不能继续称作完全没有健康先验的自监督预测模型。

## 对照

- 相同单层 GRU，8 维 hidden，504 参数；
- 相同三个 seed：20260916、20260917、20260918；每个 seed 的两臂初始化相同；
- control：仅 next-step prediction MSE；
- ranking：MSE + 0.1 × ranking loss；
- 保持 hidden-distance Level、前15点参考、10点平滑、Adam、学习率0.001、300轮预算及其余原训练设置；
- 两臂都仅按 Bearing1_4 的 next-step MSE 选 checkpoint；排序 loss 不参与验证模型选择。

定义 d_t = ||h_t - mean(h_0,...,h_14)||，L_t 为 d 的10点因果移动平均，只使用完整窗口。训练辅助损失为：

`mean_bearing mean_t relu(0.05 - (L_(t+10) - L_t))`

每个训练 bearing 的辅助损失等权平均。时间索引、总寿命和 normalized lifetime 不作为网络输入或回归目标；排序标签来源于训练序列的先后顺序。推理不强制 cumulative-max，因此真实回落仍可输出。

## 数据边界

训练只用 Bearing1_1–1_3，标准化只拟合这些 bearing；Bearing1_4 选择 checkpoint，Bearing1_5 是已经多次查看的旧留出设备。全部 checkpoint 锁定后才统一评估。前15点仅用于当前设备校准，之后的输出只依赖当前及过去观测。本实验不读取 Bearing2_5，也不声称新盲测。

## 统一评价

从零起算 step 25 评分；重新计算简单距离基线，避免混用历史不同时间区间的ρ。

- 逐 seed/bearing 的ρ、回落比例、step标准差、动态范围和 prediction MSE；
- 三个seed的ρ均值、样本标准差、最小值；
- 报告原始和按轨迹动态范围归一化的平滑度，防止缩小输出尺度造成假改善；
- 不把不同seed当作独立设备，不进行显著性推断；
- control checkpoint参数、最佳epoch、验证MSE应精确复现既有三个checkpoint。

## 防止只学习时钟

每个bearing保留前15点，之后分别输入：（1）相同的早期参考向量；（2）固定随机排列的原观测。只用于诊断，不参与训练、checkpoint选择或参数调整。

恒定输入检查同时报告ρ和绝对/相对变化幅度；数值噪声上的高ρ不足以认定时钟行为。若恒定输入的Level动态范围超过真实轨迹的10%，标记需要核查。另列纯时间计数器，其ρ必然为1，用来提醒：本轮用时间顺序训练后，高时间ρ不能单独证明健康语义。这些检查只能筛查明显捷径，不能证明不存在捷径。

## 预设推进门槛

只有 Bearing1_4 和 Bearing1_5 分别满足以下条件，才支持进入更大规模验证：seed标准差至少降低20%；平均ρ下降不超过0.01；最差seed ρ不降低；prediction MSE均值增加不超过10%；无塌缩，且恒定输入检查无超过10%的动态范围标记。该门槛是探索性决策规则，不是统计显著性检验。

失败则保留距离基线，记录失败方式，不在本轮调权重、挑seed或换checkpoint指标。成功也只表明这个假设值得继续研究，下一阶段仍需未参与选择的bearing及独立于时间顺序的评价。

## 执行与核验

`python -B scripts/run_gru_temporal_ranking.py --config configs/xjtu_gru_temporal_ranking_v1.json`

CPU单线程、确定性算法，30分钟硬时限，逐50轮报告进展。保存全部训练曲线、六个checkpoint、逐点状态、诊断指标、数据/配置/源代码/模型哈希和环境信息。输出目录存在时拒绝覆盖。协议与配置在执行后保持原样。
