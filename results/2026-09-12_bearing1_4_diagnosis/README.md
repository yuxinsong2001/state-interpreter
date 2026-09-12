# Bearing1_4 事后诊断

本实验解释为什么`Bearing1_4`是v0.1稳定性实验中最弱的bearing。它属于事后分析，不用于重新调整v0.1参数。

## 诊断结论

1. **不是随机初始化主导。** 三个seed的Level轨迹平均Pearson相关为`0.991`，说明不同AutoEncoder初始化产生了非常相似的时间形状。
2. **存在明显的Encoder/数据分布偏差。** `Bearing1_4`平均重建MSE为`0.598`，其他bearing平均为`0.338`，比例为`1.77`。
3. **校准长度比平滑窗口影响更明显。** 15/10锁定参数的Level平均ρ为`0.860`。事后延长校准可提高ρ，但会减少早期可用状态，并降低Trend的及时性，不能据此改写锁定参数。
4. **Movement仍不稳定。** 16组校准/窗口组合下，Movement与Level变化的相关性总体较弱，改变窗口不能根治问题。

因此，下一种改进应优先处理embedding尺度与个体基线的稳健性，例如稳健中心、协方差归一化距离或基于训练健康原型的相对表示；不建议仅为该bearing增加窗口，也没有证据表明应立即使用GRU或Attention。

## 文件

- `reconstruction_by_bearing.csv`：三个seed下所有bearing的重建误差。
- `seed_trajectory_agreement.csv`：三个seed的Level轨迹两两一致性。
- `window_sensitivity.csv`：4个校准长度×4个窗口×3个seed的事后结果。
- `diagnostic_report.json`：机器可读汇总。
