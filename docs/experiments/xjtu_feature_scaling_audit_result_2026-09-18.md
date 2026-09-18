# Condition 3 Feature LSTM稳定缩放审计（2026-09-18）

## Material Passport

- Amendment ID: `xjtu_condition3_feature_scaling_amendment_v1`
- Verification status: `VERIFIED`
- Data scope: 仅B3_1–B3_3既有特征缓存
- Protected data: B3_4/B3_5未读取
- Model training: 未执行

## 问题

原始早期z-score在近常量特征上产生极端值：B3_1最大绝对值约`1.18×10^10`。直接训练会让少数特征主导梯度，因此必须先建立不依赖模型结果的数值质量门。

## 预注册候选与门槛

在运行审计前固定三个候选：原始z-score、`signed_log1p`和±20硬裁剪。候选必须在三个开发bearing上同时满足：全部有限、最大绝对值≤25、绝对值p99≤20，并保持严格单调变换。选择规则为按预注册顺序选取第一个全部通过者。

## 结果

| 方法 | B3_1 max / p99 | B3_2 max / p99 | B3_3 max / p99 | 是否通过 |
|---|---:|---:|---:|---|
| 原始z-score | 1.18e10 / 2.96e8 | 2259.80 / 38.73 | 1616.95 / 198.37 | 否 |
| signed_log1p | 23.19 / 19.51 | 7.72 / 3.68 | 7.39 / 5.30 | 是 |
| ±20裁剪 | 20 / 20 | 20 / 20 | 20 / 20 | 否：非严格单调 |

锁定方案为：

\[
z=\frac{x-\mu_{early}}{\max(\sigma_{early},10^{-6})},\qquad
\tilde z=\operatorname{sign}(z)\log(1+|z|)
\]

该变换不会改变单个特征内部的数值顺序和符号，也不会像硬裁剪一样制造大面积并列值。它解决的是数值稳定性，不保证提高RUL预测或State Interpreter性能。

## 完整性与边界

- 三个输入NPZ哈希与预注册值一致。
- 稳定缩放代码哈希与预注册值一致。
- 157项完整仓库测试通过。
- B3_4/B3_5未读取。
- 没有使用RUL预测结果选择缩放方式。

## 下一步

建立使用`signed_log1p`的Feature LSTM训练协议v2，保持原三折LOBO、三个seed、50 epochs、固定最终epoch和原验证门槛不变。协议冻结后才允许开始训练。
