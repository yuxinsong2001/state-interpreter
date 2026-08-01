# RelativeTemporalStateInterpreter v1

## 目的

将有序 latent `z_t` 在线转换为三个可解释状态分量：

```text
state_t = [level_t, trend_t, movement_t]
```

该模块不依赖 AutoEncoder 内部结构，未来可以直接把输入替换为 VibFM embedding。

## 生命周期

```text
reset()
→ CALIBRATING
→ 收集固定数量的早期 z
→ 建立该 episode 自身 baseline
→ READY
→ 每个新 z 输出正式 state
```

校准阶段 `update(z)` 返回 `None`，避免调用方误把尚未建立参照系的数值当作健康状态。完成校准的最后一个样本仍不输出；从下一个样本开始进入正式在线阶段。

## 状态定义

### Level

```text
raw_level_t = ||z_t - baseline_episode||₂
level_t = mean(raw_level over current and previous window)
```

表达设备相对自身早期状态的总体偏移。

### Trend

```text
trend_t = OLS slope(recent level values)
```

表达近期 level 上升或下降的速度。正值表示偏移正在加速，负值表示近期回落。

### Movement

```text
movement_t = ||z_t - z_(t-1)||₂
```

表达相邻测量之间的突然变化，防止平滑 level 掩盖突发失效。

## 接口

```python
from state_interpreter import RelativeTemporalStateInterpreter

interpreter = RelativeTemporalStateInterpreter(
    embedding_dim=8,
    calibration_steps=10,
    temporal_window=5,
)

interpreter.reset()
for z_t in ordered_embeddings:
    output = interpreter.update(z_t)
    if output is None:
        continue
    state_t = output.state  # [level, trend, movement]
```

模块一次只接收一个 episode 的一个一维 embedding。切换 bearing、设备或 episode 前必须调用 `reset()`，或者为每个并行 episode 创建独立实例。

## 软件边界

- `embedding_dim`、`calibration_steps` 必须为正。
- `temporal_window` 必须大于1。
- 输入必须为固定维度的一维浮点 tensor。
- NaN、Inf、错误 shape 和整数输入会被拒绝。
- baseline 属性返回防御性副本，外部代码不能修改内部校准状态。
- 输入会被 detach 和 clone，模块不会保存训练计算图。
- `reset()` 清除 baseline、上一时刻 latent 和全部窗口历史。

## 测试结果

完整仓库测试：37项全部通过。新增测试覆盖：

- `CALIBRATING → READY`；
- 校准期禁止正式输出；
- level、trend、movement 数值；
- `reset()` 的 episode 隔离；
- baseline 防外部篡改；
- shape、NaN/Inf、dtype 与配置错误。

## XJTU-SY 真实在线重放

设置：

- 输入：上一轮提取的616个 `z=8`；
- 每个 bearing 固定前10个测量点用于校准；
- 每个测量间隔1分钟；
- temporal window：5；
- 五个 bearings 总计输出566个 READY states。

| Bearing | Split | READY states | Level 与 normalized lifetime 的 Spearman ρ |
|---|---|---:|---:|
| Bearing1_1 | train | 113 | 0.963 |
| Bearing1_2 | train | 151 | 0.994 |
| Bearing1_3 | train | 148 | 0.957 |
| Bearing1_4 | validation | 112 | 0.674 |
| Bearing1_5 | holdout | 42 | 0.733 |

五个 bearing 均保持正趋势。`Bearing1_4` 的末端 movement 仍保留约9.0的突变。

## 当前能说明什么

- 校准式 State Interpreter 已从离线公式变成可调用、可重置、只使用历史信息的正式模块。
- 固定10步校准在 validation 和 holdout 上仍得到正向 level 趋势。
- 三维状态能够同时表达总体偏移、近期速度和突然变化。

## 当前不能说明什么

- 不能证明 `[level, trend, movement]` 就是真实物理健康状态。
- 10步校准和5点窗口尚未通过敏感性分析确定。
- 当前只使用一个工况，不能证明跨工况、跨机器泛化。
- normalized lifetime 仍是时间代理，不是真实 damage 标签。
- 当前模块要求同一设备具有连续历史，不能直接处理完全独立、无历史的单次测量。

## 下一步

对 `calibration_steps` 和 `temporal_window` 做预先定义的敏感性实验，例如：

```text
calibration_steps ∈ {5, 10, 15, 20}
temporal_window   ∈ {3, 5, 10}
```

使用 validation bearing 选择稳健区域，holdout bearing 只做一次最终检验，避免针对 `Bearing1_5` 调参。

