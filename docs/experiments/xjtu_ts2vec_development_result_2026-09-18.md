# XJTU-SY Condition 3 紧凑 TS2Vec 开发实验结果

## 结论

本实验未通过预注册的健康排序与 bearing identity 双重门槛，因此不允许读取 `Bearing3_4` 做验证，也不允许进入 `Bearing3_5` 盲测。

TS2Vec 在 `Bearing3_1` 上得到稳定的正向健康排序，但在 `Bearing3_2` 和 `Bearing3_3` 上出现跨随机种子一致的负向排序。与此同时，bearing identity probe 的平均准确率为 0.898，仍明显高于预注册上限 0.80，并且只比原始特征的 0.903 略低。这说明当前多尺度时间对比表示仍强烈保留设备身份，没有形成跨 bearing 方向一致的共享健康坐标。

## Material Passport

- 数据：XJTU-SY Condition 3 的 `Bearing3_1`–`Bearing3_3` 开发缓存。
- 数据单位：每个 bearing 的按时间排序测量序列。
- 受保护数据：`Bearing3_4` 和 `Bearing3_5` 均未读取。
- 上游方法：官方 TS2Vec，固定 commit `b0088e14a99706c05451316dc6db8d3da9351163`，MIT License。
- 配置：`configs/xjtu_condition3_ts2vec_development_v1.json`。
- 配置 SHA-256：`31d480f1501a31f47e9419d3f9594eb436b5914822f5878de7713316360dc3ad`。
- 实验状态：`ANALYZED`。结果已执行并分析，但未做第二次独立复跑，因此不标记为 `VERIFIED`。

## 固定方法

- 外层评价：`Bearing3_1`–`Bearing3_3` 三折 leave-one-bearing-out。
- 输入缩放：只根据每个 bearing 最初 15 步校准的 `signed_log1p` 特征。
- 训练样本：每个训练 bearing 等距抽取 16 个长度为 128 的片段。
- Encoder：65 维输入、64 维输出、hidden 32、depth 6。
- 训练：AdamW，学习率 0.001，固定 200 iterations，三个预注册 seed。
- 推理：仅使用当前及过去最多 128 步，取最后 timestamp representation；不使用未来上下文。
- 健康读出：仅在外层训练 bearings 上拟合 Ridge，目标为 normalized lifetime。
- 身份探针：nearest centroid，五个连续生命周期区段。

## 结果

| 外层测试 bearing | Mean Spearman ρ | Std | 正相关 seeds |
|---|---:|---:|---:|
| Bearing3_1 | 0.598 | 0.096 | 3/3 |
| Bearing3_2 | -0.452 | 0.154 | 0/3 |
| Bearing3_3 | -0.330 | 0.087 | 0/3 |

单 seed 结果：

- `Bearing3_1`：0.645、0.465、0.685。
- `Bearing3_2`：−0.656、−0.283、−0.416。
- `Bearing3_3`：−0.431、−0.341、−0.218。

Identity probe：

- 平均准确率：0.898。
- 允许上限：0.80。
- 原始特征参考：0.903。

门槛判定：

- 三个 bearing 的平均相关性全部为正：失败。
- 至少两个 bearing 的平均相关性不低于 0.5：失败。
- 每个 bearing 至少两个 seed 为正：失败。
- Identity 平均准确率不高于 0.80：失败。
- 联合门槛：失败。

## 解释

结果不支持“只要换成成熟的通用时间对比模型，就能自动获得共享健康表示”。TS2Vec 能在单个 bearing 上捕捉时间结构，但它的 instance contrast 并不显式要求不同 bearing 的退化方向对齐。当前结果与这一局限一致：模型在 `Bearing3_1` 上获得了较强正相关，却在另两个 bearing 上稳定地学习到相反方向，并几乎完整保留 bearing identity。

这不是简单的训练随机性：`Bearing3_2` 和 `Bearing3_3` 的三个 seed 全部为负。因此不应通过挑选 seed、增加 iterations、降低门槛或查看 `Bearing3_4` 来修饰结果。

## 研究有效性检查

- 分 bearing 报告结果，避免用总体平均掩盖方向冲突。
- bearing 是泛化单位，测量点不能被当作独立设备样本。
- 采用预注册配置，避免结果后调参和多重尝试偏差。
- 测试 bearing 不参与 checkpoint 选择或 Ridge 拟合。
- 仅进行相关性解释，不声称因果或真实物理损伤量。
- 该结论限于 XJTU-SY Condition 3 和当前紧凑适配。

## 下一步

停止继续微调纯 TS2Vec。下一种候选应在时间目标之外显式加入跨 bearing/domain 对齐，例如 DANN、MMD 或其他 latent alignment，并继续沿用相同的 LOBO 与 identity 双门槛。只有开发集门槛通过后，才允许读取 `Bearing3_4`。

