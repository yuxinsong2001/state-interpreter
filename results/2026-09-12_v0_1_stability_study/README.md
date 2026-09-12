# v0.1 稳定性实验

本目录记录 `AutoEncoder + RelativeTemporalStateInterpreter` 在两个工况上的多随机种子、跨 bearing 稳定性实验。

## 实验设计

- 工况：`35Hz12kN`、`37.5Hz11kN`
- 每个工况：5折 bearing 轮换测试
- 每折：3个 bearing 训练、1个 bearing 验证、1个 bearing 测试
- 随机种子：`20260801`、`20260802`、`20260803`
- 总训练次数：30
- AutoEncoder：`z=8`，5 epochs
- State Interpreter：15步校准、10点时间窗口

归一化统计量只在每折训练 bearings 上拟合，测试 bearing 不参与模型训练、归一化或 checkpoint 选择。

## 主要结果

| 范围 | Level–lifetime Spearman ρ（均值±标准差） | 95% CI |
|---|---:|---:|
| 全部30次 | 0.959 ± 0.048 | [0.941, 0.977] |
| 35Hz12kN | 0.940 ± 0.056 | [0.909, 0.971] |
| 37.5Hz11kN | 0.978 ± 0.029 | [0.962, 0.994] |

- Trend方向一致率：`0.816 ± 0.075`。
- Movement与Level绝对变化的相关性：`ρ=0.379 ± 0.206`。
- 最弱的bearing组合是`Bearing1_4`，三个种子的Level平均`ρ≈0.86`。
- Level在不同随机种子、bearing和两个工况下总体稳定。
- Trend具有一定方向解释能力，但仍有噪声。
- Movement只表现出中等且波动较大的关联，尚不能作为可靠退化指标。

## 结论边界

该实验支持第一版`Level`作为稳定的相对退化排序指标，但不证明它等于物理损伤或RUL。Trend和Movement当前只验证了内部时间一致性，尚未验证对维护决策或RL的实际价值。30次运行中同一折的不同随机种子共享数据，因此不能被视为完全独立样本。

## 文件

- `per_run_metrics.csv`：30次运行的逐次结果。
- `aggregate_metrics.csv`：总体及分工况均值、标准差和95%置信区间。
- `study_report.json`：机器可读实验协议、参数、结果及限制。

## 后续方法选择

保留当前基于早期基线距离的Level。下一轮优先针对`Bearing1_4`分析失败原因，并研究更稳健的Trend和Movement定义；在证据表明固定窗口不足后，再考虑HMM、GRU或Attention等时序模型。
