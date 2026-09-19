# Condition 3 条件 MMD：训练样本覆盖预检

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + read-only preflight
- Origin Date: 2026-09-19
- Verification Status: ANALYZED（只核对缓存清单、哈希与 endpoint 分段；没有验证 MMD 数值）
- Version Label: source_conditional_mmd_metadata_preflight_v1
- 输入：`cache/xjtu_condition3_feature_lstm_v1/manifest.json`、三份已有 `Bearing3_1`–`Bearing3_3` cache 文件、[实验协议草案](xjtu_source_conditional_mmd_protocol_2026-09-19.md)。
- 访问边界：未读取 `Bearing3_4` / `Bearing3_5`，未读取健康预测结果、未训练、未修改已冻结配置。

## 检查方法

按现有 `evenly_spaced_indices(length, 300)` 规则，以每个训练 bearing 的 300 个 endpoint 构建 `endpoint / (length - 1)` 时间进度；区间索引为 `min(4, floor(progress × 5))`。这与 Source-DANN 的 endpoint 生成方式一致。读取已有 manifest 中的样本数，检查三份 cache 的 SHA-256 是否与 manifest 一致；没有访问受保护 bearing。

## 结果

| Bearing | 缓存样本数 | 0–20% | 20–40% | 40–60% | 60–80% | 80–100% | 缓存哈希 |
|---|---:|---:|---:|---:|---:|---:|---|
| B3_1 | 2538 | 60 | 60 | 60 | 60 | 60 | 匹配 |
| B3_2 | 2496 | 60 | 60 | 60 | 60 | 60 | 匹配 |
| B3_3 | 371 | 60 | 60 | 59 | 60 | 61 | 匹配 |

三折 LOBO 的任意两个训练 bearing 在每个区间都有至少 59 个 endpoint。因此，“某区间无配对样本”这个停止条件目前没有触发。**但原有 batch_size=64 的普通随机 batch 不能保证每个 batch 同时包含两个 bearing 的全部五个区间；条件 MMD 需要专门的配对采样或整批损失。** B3_3 还存在 59/61 的小幅不平衡；不能声称 300 个 endpoint 可以无替换地拆成每区间完全相同的 60 个样本。

## 尚未通过的门槛

1. 核函数、带宽估计和 MMD 估计量尚未锁定；本步没有计算 embedding 间距离或损失值。
2. 尚未比较健康 MSE 与 global/conditional MMD 的梯度尺度，正则权重不能据此冻结。
3. 尚未验证分层配对采样能在三个分支上保持相同输入集合、处理最后一个 batch 且不引入替换偏差。
4. 尚未建立代码层 fail-closed 检查、单元测试和独立配置哈希。

结论：**数据覆盖预检通过，完整数值预检未完成，训练仍不得启动。**下一步先设计训练来源限定的分层配对采样和核尺度诊断，再执行数值测试；任何权重和带宽只允许根据训练 bearing 的数据及预先规定的稳定性标准确定，不准根据 LOBO 留出结果调优。

## 解释限制

五等分是测量时间代理，不是物理故障阶段。缓存文件哈希匹配只能保证输入与既有 manifest 一致，不能证明特征适合健康表征。这里的 59–61 是 endpoint 分布，不是独立实验样本数；同一 bearing 内时间点强相关。
