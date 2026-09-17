# 相对 State Interpreter v2：集成结果（2026-09-17）

## 完成结果

已形成统一在线状态接口：

```text
z_t
├─ signed degradation axis → Level
├─ causal Level slope      → Trend
└─ frozen Residual-GRU prediction surprise → Movement
```

前15个embedding只用于当前bearing校准，此后每个时间步输出 `[Level, Trend, Movement]`。三个Residual-GRU checkpoint采用集成平均，Movement以训练bearing典型预测偏差 `0.059263` 为尺度，因此训练数据总体中位Movement约为1。

## 评价结果

| Bearing | Level–寿命ρ | Trend–寿命ρ | Movement–寿命ρ | Movement中位数 | Movement P95 |
|---|---:|---:|---:|---:|---:|
| B1_1 | 0.952 | -0.082 | -0.725 | 1.272 | 8.713 |
| B1_2 | 0.996 | -0.504 | -0.804 | 0.600 | 10.264 |
| B1_3 | 0.959 | 0.705 | -0.102 | 1.181 | 3.813 |
| B1_4 | 0.927 | -0.094 | 0.252 | 1.599 | 5.005 |
| B1_5 | 0.990 | 0.910 | -0.008 | 1.221 | 8.931 |

Level稳定保持生命周期排序能力。Trend表达的是局部Level方向，在平台、回落和加速阶段会改变正负，因此其全生命周期相关性不应作为主要性能指标。Movement同样不单调：它表达模型对当前变化的意外程度，而不是损伤程度。

B1_4最后一点Movement约88，说明该点远离模型根据历史预测的下一状态；这可以作为异常/转移候选，但在没有故障事件标签时不能断言它就是失效时刻。B1_5在约0.65–0.80生命周期出现多个Movement峰值，之后回落，展示了它与Level不同的局部变化含义。

## 完整性验证

- 已归档signed-axis Level最大复现误差低于 `1.7e-7`。
- 所有bearing前缀执行与完整执行一致。
- 所有状态有限，Movement全部非负。
- 训练Movement中位数为 `1.00000075`。
- 输入、axis和三个checkpoint哈希保持不变。
- 全仓库121项测试通过。

## 结论边界

当前版本已经是一套逻辑闭合的**无标签相对State Interpreter**：Level回答相对偏离程度，Trend回答近期方向，Movement回答当前变化是否超出模型预期。它尚不能输出跨bearing统一损伤百分比、物理退化阶段或RUL。

下一步应验证Movement峰值是否对应可解释事件：先定义不使用评价bearing调参的训练阈值，再检查峰值的持续时间、与Level/Trend转折的关系以及不同seed的一致性。没有事件标签时，只能称为change-point候选。

