# HMM Log-domain Filtering Check

本目录保存第一版HMM从概率域过滤改为log-domain过滤后的数值验证结果。除过滤计算域和输出目录外，输入latent、HMM拟合参数、距离基线和评分区间均保持不变。

与`results/2026-09-16_hmm_interpreter_exploratory/`逐时间步比较：

- 616个时间步的离散阶段不一致数：0；
- `hmm_probability_0/1/2`最大绝对差：0；
- `hmm_expected_stage`最大绝对差：0；
- `hmm_confidence`最大绝对差：0；
- `hmm_parameters.json` SHA-256完全一致。

因此，v1中的过度置信与Bearing1_4恒定状态不是概率域下溢造成的。log-domain实现仍被保留，因为它能安全处理极端embedding，并通过了相应单元测试。
