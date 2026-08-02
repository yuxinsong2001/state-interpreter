## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-02
- Verification Status: UNVERIFIED
- Version Label: exp_result_v1

# State Interpreter 参数敏感性实验

## Experiment Result

- **ID**: `interpreter_parameter_sweep_z8_20260802`
- **Type**: analysis
- **Status**: completed
- **Input**: `runs/latent_analysis_z8_20260801/latent_trajectories.csv`
- **Output**: `runs/interpreter_parameter_sweep_z8_20260802/`
- **Exit Code**: 0
- **执行说明**：本次实现、测试、扫描和参数整理由 Codex 在用户授权下完成，不表示用户已经亲自完成或理解了全部步骤。

## 实验问题

`RelativeTemporalStateInterpreter` 对校准长度和时间窗口是否敏感？应当在不使用 holdout 的前提下锁定哪一组第一版参数？

## 参数网格

```text
calibration_steps ∈ {5, 10, 15, 20}
temporal_window   ∈ {3, 5, 10}
```

共12组配置。

## 数据边界

- 训练参考：`Bearing1_1–Bearing1_3`。
- 参数选择：`Bearing1_4`。
- Holdout：`Bearing1_5`，扫描时明确排除。
- 报告字段：`excluded_bearings=[Bearing1_5]`。
- 报告字段：`holdout_metrics_computed=false`。

## 预先固定的排序规则

1. `Bearing1_4` 的 level–normalized-lifetime Spearman ρ 降序；
2. validation smoothness 降序；
3. 完整窗口可用时间升序。

同时报告 directional monotonicity、trend noise、READY 样本比例和 movement 突变保留。不同指标不被任意加权成单个总分。

## 完整排名

| Rank | Calibration | Window | Validation ρ | Smoothness | Trend noise | READY比例 | 完整窗口时刻 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 10 | 0.865 | 0.626 | 0.021 | 0.836 | 29 |
| 2 | 20 | 5 | 0.864 | 0.568 | 0.049 | 0.836 | 24 |
| 3 | 20 | 3 | 0.863 | 0.521 | 0.168 | 0.836 | 22 |
| 4 | 15 | 10 | 0.849 | 0.620 | **0.012** | 0.877 | 24 |
| 5 | 15 | 5 | 0.843 | 0.568 | 0.044 | 0.877 | 19 |
| 6 | 15 | 3 | 0.838 | 0.519 | 0.162 | 0.877 | 17 |
| 7 | 10 | 10 | 0.681 | 0.636 | 0.012 | 0.918 | 19 |
| 8 | 10 | 5 | 0.674 | 0.579 | 0.043 | 0.918 | 14 |
| 9 | 10 | 3 | 0.668 | 0.521 | 0.158 | 0.918 | 12 |
| 10 | 5 | 10 | 0.613 | **0.649** | 0.015 | 0.959 | 14 |
| 11 | 5 | 5 | 0.602 | 0.588 | 0.043 | 0.959 | 9 |
| 12 | 5 | 3 | 0.592 | 0.526 | 0.154 | 0.959 | 7 |

## 主要发现

### 校准长度影响最大

在相同窗口下，validation ρ 随校准长度从5增加到20而明显提高：

```text
约0.59–0.61
→ 约0.67–0.68
→ 约0.84–0.85
→ 约0.863–0.865
```

说明 `Bearing1_4` 需要比最初10步更长的早期观察，才能得到稳定个体 baseline。

### 时间窗口主要控制平滑和趋势噪声

同一校准长度下，窗口从3增加到10：

- validation ρ 只小幅提高；
- smoothness 明显提高；
- trend noise 明显下降；
- 完整窗口状态更晚可用。

`movement` 不进行窗口平滑，因此12组配置都保留 `Bearing1_4` 末端约9.007的突变。

### 存在稳定区域，而非单点尖峰

校准15–20步的六组配置全部位于最佳 validation ρ 的0.03以内，说明当前方法在该区域相对稳定。校准5–10步则形成明显较弱区域。

## 参数锁定决策

第一版锁定：

```text
calibration_steps = 15
temporal_window = 10
```

没有直接选择排名第一的 `20/10`，原因是：

- `15/10` 的 validation ρ=0.849，仅比 `20/10` 低0.016；
- 少等待5个校准测量；
- READY 样本比例从0.836提高到0.877；
- smoothness 0.620，与0.626接近；
- trend noise 0.012，低于 `20/10` 的0.021；
- 训练 bearings 平均ρ为0.971，高于 `20/10` 的0.966。

版本化配置：`configs/xjtu_z8_interpreter_v1.json`。

## 当前门槛

- **通过**：参数敏感性扫描和 validation-only 参数锁定。
- **未执行**：锁定参数后的 `Bearing1_5` 单次 holdout 评价。
- **未通过**：真实物理健康有效性、跨工况泛化和维护决策有效性。

## 生成工件

- `sweep_results.csv`：12组排名与汇总指标。
- `per_bearing_metrics.csv`：48组逐 bearing 指标。
- `sweep_report.json`：数据边界、排序规则和近最优配置。
- `validation_spearman_heatmap.png`：validation 趋势热力图。
- `validation_smoothness_heatmap.png`：validation 平滑性热力图。

## 下一步

读取锁定配置，对 `Bearing1_5` 只运行一次最终 holdout 重放。运行后无论结果好坏，都先报告，不再根据 holdout 修改参数。

