# v0.1 多随机种子、跨 bearing 与跨工况稳定性实验

## 研究问题

正式盲测只使用一个未知bearing，仍需检查高相关性是否依赖特定数据划分或随机初始化。本实验保持`z=8` AutoEncoder结构和15/10 State Interpreter不变，系统评估其稳定性。

## 协议

- 工况：`35Hz12kN`与`37.5Hz11kN`。
- 每个工况5折：一个测试bearing、循环中的下一个验证bearing、其余三个训练bearing。
- 种子：`20260801`、`20260802`、`20260803`。
- 合计：2工况 × 5折 × 3种子 = 30次训练。
- 归一化只在当前折的训练bearing上拟合。
- 验证bearing只用于选择5个epoch中的最佳checkpoint。

## 结果

| 范围 | Level Spearman ρ均值 | 标准差 | 95% CI |
|---|---:|---:|---:|
| 全部 | 0.959 | 0.048 | [0.941, 0.977] |
| 35Hz12kN | 0.940 | 0.056 | [0.909, 0.971] |
| 37.5Hz11kN | 0.978 | 0.029 | [0.962, 0.994] |

Trend方向一致率为`0.816 ± 0.075`。Movement与Level绝对变化的相关性为`0.379 ± 0.206`。最弱对象是`Bearing1_4`，三个种子的Level平均ρ约为0.86。

## 判断

当前证据支持保留Level：它对随机种子、bearing及两个工况较稳定。Trend存在一定有效信号，但需要降低噪声。Movement表现出较强的bearing和seed敏感性，不能仅凭当前定义称为稳定退化指标。

下一轮不应直接替换整个State Interpreter。应首先分析`Bearing1_4`，并尝试更稳健的Trend与Movement定义。只有在固定窗口方法确实不足时，再比较HMM、GRU或Attention。

## 限制

normalized lifetime是时间代理而不是物理损伤标签。Trend和Movement指标衡量内部时间一致性，不等于维护决策价值。同一折的三个seed共享数据，所以30行结果不是30个完全独立的统计样本。

完整工件位于`results/2026-09-12_v0_1_stability_study/`。
