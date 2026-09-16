# HMM Interpreter Exploratory Run

本目录保存2026-09-16第一版三状态左到右高斯HMM与距离基线的探索性比较。

- `per_step_states.csv`：逐时间步的距离Level、HMM阶段概率、期望阶段和置信度；
- `bearing_metrics.csv`：在统一评分区间上的逐bearing指标；
- `hmm_parameters.json`：标准化、发射分布、转移矩阵和EM记录；
- `experiment_report.json`：输入/config哈希、完整指标和证据边界。

核心结果：HMM降低了短期回摆，但置信度接近1，并在Bearing1_4上塌缩为恒定状态，不能认为优于距离基线。完整解释见`docs/experiments/hmm_state_interpreter_result_2026-09-16.md`。

注意：报告中的“Display normalization ... plot”提示来自移除可选绘图依赖前的旧文本；本目录没有生成图像，该文本不参与任何指标计算。
