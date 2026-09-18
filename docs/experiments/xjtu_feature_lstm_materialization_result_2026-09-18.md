# XJTU-SY Feature LSTM 开发特征物化结果（2026-09-18）

## Material Passport

- Experiment ID: `xjtu_condition3_feature_lstm_development_v1`
- Artifact type: development-only feature cache
- Verification status: `VERIFIED`
- Input scope: `Bearing3_1`、`Bearing3_2`、`Bearing3_3`
- Protected scope: `Bearing3_4`、`Bearing3_5` 未读取
- Training performed: 否

## 目的

按预注册协议将 Condition 3 的三个开发 bearing 转换为固定的 65 维工程特征缓存，为后续三折 LOBO Feature LSTM 实验准备可复用输入。该步骤只验证数据完整性和数值质量，不评价模型性能。

## 执行保护

- 在访问数据前验证预注册配置、一次性令牌和三个冻结代码工件的 SHA256。
- 程序只拼接三个明确授权的 bearing 路径，不枚举 `40Hz10kN` 父目录。
- 每个 bearing 独立生成 NPZ，字段固定为 `features`、`step_ids` 和 `source_paths`。
- 缓存目录已加入 `.gitignore`；Git 只保存程序、测试和结果说明，不保存大体积数据缓存。
- 已完成独立 SHA256 和 NPZ shape/finite 复核。

## 结果

| Bearing | 测量数 | 特征形状 | 时间编号 | 有限值 | 早期近常量特征数 |
|---|---:|---:|---|---|---:|
| Bearing3_1 | 2538 | 2538 × 65 | 0–2537，连续唯一 | 通过 | 2 |
| Bearing3_2 | 2496 | 2496 × 65 | 0–2495，连续唯一 | 通过 | 1 |
| Bearing3_3 | 371 | 371 × 65 | 0–370，连续唯一 | 通过 | 0 |

三个 NPZ 的独立哈希复核均通过。全部 5405 个测量已物化，所有特征均为有限值，完整生命周期内没有零方差特征。

## 训练前质量风险

当前固定早期校准使用前 15 点的 feature-wise 标准差，并将最小值截断为 `1e-6`。B3_1 的两个主频特征在早期窗口近似常量，但后期发生离散变化，因而标准化后的最大绝对值达到约 `1.18 × 10^10`；B3_2 最大值约 `2.26 × 10^3`，B3_3 最大值约 `1.62 × 10^3`。

这不是 NaN 或读取错误，而是“早期方差过小 + 后期频率档位改变”造成的数值放大。若直接训练，少数特征可能主导梯度，使 Feature LSTM 的结果不再代表原定的 65 维基线。

## 决策

当前缓存本身通过完整性检查，但训练质量门暂不通过。因此本步骤停止在“已物化、未训练”。下一步应在不访问 B3_4/B3_5 的前提下，仅用 B3_1–B3_3 比较预先定义的稳定缩放方案，例如：

1. 对近常量校准维度禁用标准化并保留原始中心化值；
2. 为每个特征设置基于开发训练 bearing 的尺度下限；
3. 对标准化值采用预先固定的稳健裁剪。

选定规则后必须形成协议修订并重新冻结校准工件，再开始原计划的三折 LOBO；不得根据 B3_4/B3_5 选择缩放方案。

## 验证

- 定向协议与物化测试：9 passed
- 完整仓库回归：152 passed
- NPZ 哈希、形状与有限值独立复核：全部通过
- B3_4/B3_5：未读取
