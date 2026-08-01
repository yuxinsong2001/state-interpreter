## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-01
- Verification Status: UNVERIFIED
- Version Label: exp_result_v1

# `z=8` Health Indicator 候选比较

## Experiment Result

- **ID**: `health_indicator_comparison_z8_20260801`
- **Type**: analysis
- **Status**: completed
- **Input**: `runs/latent_analysis_z8_20260801/latent_trajectories.csv`
- **Output**: `runs/health_indicator_comparison_z8_20260801/`
- **Exit Code**: 0
- **执行说明**：本次实现、测试和运行由 Codex 在用户授权下完成，不表示用户已经亲自完成或理解了全部步骤。

## 比较对象

### Candidate A：全局健康中心距离

```text
global_distance_t = ||z_t - global_training_healthy_center||₂
```

用途：保留为最简单的跨 bearing 对照。

### Candidate B：个体早期基线相对距离

```text
baseline_b = mean(first 10% z of bearing b)
self_relative_distance_t = ||z_t - baseline_b||₂
```

用途：消除不同 bearing 的初始 latent 偏移。

### Candidate C：轻量可解释时序状态

```text
state_level_t    = causal_mean(self_relative_distance, last 5 points)
state_trend_t    = causal_slope(state_level, last 5 points)
state_movement_t = ||z_t - z_(t-1)||₂
```

输出是三维状态，不将三个含义不同的量任意加权为一个分数。

## 核心结果

| 指标 | 平均 Spearman ρ | 最差 bearing ρ | 正相关 bearing | 平均平滑性 | 最小有符号跨 bearing 趋势相关 | Prognosability |
|---|---:|---:|---:|---:|---:|---:|
| A 全局距离 | 0.550 | −0.799 | 4/5 | 0.429 | −0.836 | 0.803 |
| B 个体相对距离 | 0.901 | 0.773 | 5/5 | 0.429 | 0.507 | **0.823** |
| C 因果平滑相对距离 | **0.915** | **0.829** | **5/5** | **0.646** | **0.513** | 0.712 |

其中 smoothness 是本实验定义的粗糙度转换分数，不是统一行业标准；数值越高代表曲线越平滑。

## 逐步判断

### A 为什么不合适

全局距离在 `Bearing1_4` 上为反向趋势，并使跨 bearing 最小有符号相关降到 −0.836。它不能作为当前第一版通用 health level，但应保留为失败对照。

### B 解决了什么

以每个 bearing 自身早期状态为参考后：

- 五个 bearings 全部变成正相关；
- 最差相关从 −0.799 提高到0.773；
- 跨 bearing 轨迹方向从冲突变为一致；
- prognosability 在三者中最高。

这支持“跨 bearing 的主要问题之一是初始 latent 未对齐”。

### C 为什么仍然需要三个分量

5点因果平滑将平均相关提高到0.915、最差相关提高到0.829，平滑性提高到0.646。但平滑会延迟突变：`Bearing1_4` 最后一步的巨大变化被平均后，最终 `state_level` 只有约4.82，而原始相对距离约12.23。

因此不能只输出平滑后的 level；还必须同时保留：

- `state_trend`：近期退化速度；
- `state_movement`：单步突变，`Bearing1_4` 最后约为9.007。

## 当前架构决策

第一版 State Interpreter 采用 Candidate C：

```text
z_t
→ per-bearing early calibration
→ relative distance
→ causal smoothing + recent slope + Δz
→ state_t = [level_t, trend_t, movement_t]
```

Candidate A 与 B 继续保留在实验中作为 ablation baselines。

## 门槛判断

- **通过：进入轻量 State Interpreter 原型实现。**
- **未通过：宣称其已经表示真实健康或可直接用于维护决策。**

## 关键限制

1. 早期10%是离线假设的校准窗口；部署时必须先观察完整校准窗口，之后状态才严格在线可用。
2. 当前比较使用 normalized lifetime 作为时间代理，不是真实 damage 或 stage 标签。
3. 只有一个工况、五个 bearings，不能据此证明跨工况泛化。
4. prognosability 因平滑后末端值差异扩大而下降，Candidate C 不能仅凭平均相关被称为全面优胜。
5. smoothness 定义是本项目的工程指标，需要在论文/汇报中明确公式，不能与其他论文未加说明地直接比较。

## 生成工件

- `state_features.csv`：Candidate A/B/C 的616个逐时间点特征。
- `indicator_bearing_metrics.csv`：逐 bearing、逐 indicator 的指标。
- `comparison_report.json`：机器可读汇总。
- `indicator_comparison.png`：A/B/C level 曲线比较。
- `temporal_state_features.png`：Candidate C 的 level、trend、movement。

## 下一步

将 Candidate C 封装为一个具有显式 calibration 生命周期的 `RelativeTemporalStateInterpreter`：

```text
CALIBRATING → READY
```

并测试：

- 校准未完成时拒绝输出正式状态；
- 每个 bearing 的状态相互隔离；
- 当前输出不读取未来样本；
- `reset()` 能安全开始新 episode；
- 输出固定为 `[level, trend, movement]`。

