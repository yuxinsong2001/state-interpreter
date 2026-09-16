# 固定GRU checkpoint的状态读出对照协议

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：plan
- Origin Date：2026-09-16
- Verification Status：UNVERIFIED
- Version Label：gru_readout_comparison_protocol_v1

## 研究问题

三个GRU seed的next-step MSE接近，但hidden-distance Level在Bearing1_4和1_5上的排序不同。本实验检验：初始化敏感性主要来自GRU动力学模型，还是来自对hidden state使用欧氏距离的读出定义？

## 固定条件

- 使用已经训练完成的三个checkpoint；
- 不重新训练、不选择seed、不修改模型参数；
- 使用checkpoint中保存的训练集标准化统计量；
- bearing-specific前15点中心化保持不变；
- 10点移动平均保持不变；
- 三种读出统一从step 25评分。

## 三种因果读出

### 1. Hidden-state distance

```text
||h_t - mean(h_0...h_14)||₂
```

这是当前方法。它直接依赖GRU内部坐标。

### 2. Predicted-z distance

```text
z_hat_t = prediction made at t-1
||z_hat_t - mean(z_hat_1...z_hat_15)||₂
```

预测值位于标准化后的embedding输出空间。step t的预测只使用到t−1的信息。

### 3. One-step prediction residual

```text
||z_t - z_hat_t||₂
```

它衡量当前观测偏离GRU预测的程度，更接近变化或异常指标，不预设其应当随寿命单调上升。

## 公平时间区间

Predicted-z和residual的第一个对齐值位于step 1；使用step 1–15建立或跨过早期参考，step 16开始形成读出。为了让10点移动平均在三种方法中都完全填满，统一从step 25计算指标。

## 评价指标

- 每个seed和bearing的Spearman ρ；
- 跨seed ρ均值、样本标准差、最小值和最大值；
- Level轨迹的seed间两两Pearson和Spearman；
- collapse计数；
- step标准差和backward-step比例；
- hidden-distance与既有多seed结果的逐点复现误差。

## 决策规则

如果output-space读出在Bearing1_4和1_5上同时表现出更小seed标准差、更高最差ρ且无塌缩，则优先替换hidden-distance读出。如果prediction residual不呈单调趋势，应将其视为异常/变化指标，而不是健康Level，不因ρ较低而直接判定无用。

## 限制

- 所有bearing均为已见数据；
- 三个seed不是独立设备样本；
- 本实验只比较读出，不验证新的GRU泛化能力；
- ρ只评价时间排序；
- output-space更稳定也不等于具有物理损伤语义。
