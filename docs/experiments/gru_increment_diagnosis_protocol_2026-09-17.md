# 冻结 GRU 潜空间增量诊断方案（2026-09-17）

## 目的

上一轮结果表明，6 个冻结 GRU checkpoint 在 5 个 bearing 上均未超过 persistence baseline。本实验不训练新模型，而是检查 GRU 的一步预测误差主要来自：

1. 固定偏差（bias）；
2. 增量方向错误；
3. 增量幅度错误。

诊断结果用于判断下一步是否有证据支持 residual/increment prediction 架构，而不是直接宣布该架构有效。

## 固定对象

- 输入：与上一轮相同的 8 维 latent trajectory；
- checkpoint：control/ranking × 三个随机种子；
- bearing：Bearing1_1–Bearing1_5；
- 主分析目标：target step ≥ 25；
- 不重新训练、不选择 checkpoint、不调整超参数。

## 计算定义

对每个 origin step `t`：

- 真实增量：`Δz_true(t) = z(t+1) - z(t)`；
- 预测增量：`Δz_pred(t) = z_hat(t+1) - z(t)`；
- 预测误差：`e(t) = Δz_pred(t) - Δz_true(t)`。

检查指标：

- raw MSE 及其相对 persistence MSE 的比值；
- mean error bias 与 bias 占 raw MSE 的比例；
- 使用整段评估数据均值进行事后去偏后的 oracle MSE；
- 预测增量与真实增量的 RMS 幅度比；
- flattened cosine、through-origin gain；
- 每一步内积为正的比例以及 step cosine。

## 解释边界

- oracle bias correction 使用了整段真实目标，只能解释误差结构，不能作为部署结果；
- 标记阈值仅用于描述，不是显著性检验；
- 这些 bearing 已被查看过，Bearing1_4 参与 checkpoint 选择，Bearing1_5 也已多次分析；
- 三个 seed 只反映训练随机性，不代表设备间不确定性；
- 能预测 latent increment 不等于已经获得物理健康语义。

## 决策方式

- 若 bias 占比高且 oracle 去偏明显改善，下一步优先研究可因果估计的 bias calibration；
- 若方向总体正确但幅度明显错误，residual prediction 或增量幅度正则化具有实验依据；
- 若方向也不稳定，不能仅通过 residual 输出层解决，应先重新评估目标、序列信息和表示空间。
