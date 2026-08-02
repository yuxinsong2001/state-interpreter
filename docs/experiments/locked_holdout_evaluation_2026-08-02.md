## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-02
- Verification Status: UNVERIFIED
- Version Label: exp_result_v1

# 锁定参数下的 Bearing1_5 评价

## Experiment Result

- **ID**: `locked_holdout_bearing1_5_20260802`
- **Type**: analysis
- **Status**: completed
- **Config**: `configs/xjtu_z8_interpreter_v1.json`
- **Output**: `runs/locked_holdout_bearing1_5_20260802/`
- **Exit Code**: 0
- **执行说明**：本次脚本实现、测试、运行和结果整理由 Codex 在用户授权下完成，不表示用户已经亲自完成或理解了全部步骤。

## 评价边界

本次使用在 `Bearing1_4` 参数扫描后锁定的配置：

```text
embedding_dim = 8
calibration_steps = 15
temporal_window = 10
holdout bearing = Bearing1_5
```

运行前配置 SHA-256：

```text
9b4a9349e64f0c02a5e7fb8185f73ee4cf99950308376acb2321efeed2be8514
```

输入 latent CSV SHA-256：

```text
1fbff8151767b124b68528ea5228f6e76cc391cc9b1ead333080d68e01fb9ec0
```

脚本只评价 `Bearing1_5`，报告字段 `other_bearings_evaluated=false`。运行结束后没有修改15/10参数。

### 关于“holdout”的准确称呼

`Bearing1_5` 没有参与最近的参数敏感性扫描，因此它是**参数选择 holdout**。但它在更早的探索性 latent 分析和候选方法比较中已经被查看过，因此不是项目全过程中完全未见的严格 blind holdout。本次结果只能提供受限的独立支持，不能被描述为完全无偏的最终测试。

## 结果

| 指标 | Bearing1_5 | Validation Bearing1_4 | 差值/说明 |
|---|---:|---:|---:|
| Level Spearman ρ | **0.950** | 0.849 | +0.101 |
| Directional monotonicity | 0.556 | 0.226 | 更一致地上升 |
| Smoothness | **0.837** | 0.620 | 更平滑 |
| Trend noise | 0.034 | 0.012 | holdout trend 波动略高 |
| READY states | 37 | 107 | Bearing1_5 总寿命更短 |
| READY比例 | 0.712 | 0.877 | 15步校准占比更大 |
| 首个 level | 0.240 | — | 校准后仍接近自身 baseline |
| 最终 level | 7.142 | — | 明显偏离早期 baseline |
| 最大 movement | 1.176 | 9.007 | 退化模式不同 |

两个预先报告的描述性判断均成立：

- level 方向一致：ρ>0；
- holdout ρ 没有比 validation 低0.20以上。

## 曲线解释

- step 15–32：level 接近早期 baseline，trend 接近0。
- step 33以后：level 持续上升，trend 明显转正。
- movement 在step 34–42出现多个中等峰值，最大值约1.176，位于step 39。
- 与 `Bearing1_4` 的最后一步巨大跳变不同，`Bearing1_5` 表现为一段持续加速退化，而非单次末端断裂。

这支持三维状态的必要性：level 表达累积变化，trend 表达退化加速，movement 区分局部突变形式。

## 可以得出的结论

在以下受限条件下，锁定的第一版 State Interpreter 得到初步跨-bearing支持：

```text
XJTU-SY
35Hz12kN
z=8 AutoEncoder
15步校准
10点因果窗口
```

`Bearing1_5` 的 level 保持强正趋势，且没有针对它重新调参。

## 不能得出的结论

- 不能宣称这是完全盲测，因为早期探索阶段查看过 `Bearing1_5`。
- 不能把 normalized lifetime 当作真实 damage 标签。
- 不能证明跨工况、跨机器或跨数据集泛化。
- 不能证明状态可直接用于维护阈值或 RL 决策。
- 不能因为本次结果较好而继续用 `Bearing1_5` 调整 v1 参数。

## 生成工件

- `locked_config_snapshot.json`：评价前锁定配置快照。
- `holdout_states.csv`：52个输入的 calibration/READY 状态与37个正式输出。
- `holdout_report.json`：配置哈希、输入哈希、指标和范围声明。
- `holdout_state_plot.png`：level、trend、movement 曲线。

## 下一步建议

冻结 v1，不再针对 `Bearing1_5` 修改。下一轮应优先扩展到未用于当前开发的新证据，例如：

1. XJTU-SY 其他工况，检验工况变化是否破坏 latent 与状态解释；
2. 预先指定新的 bearing/data split，建立真正未见的 v2 blind evaluation；
3. 如果老师近期更关注模型原理，先整理当前结果并汇报，再决定跨工况实验范围。

