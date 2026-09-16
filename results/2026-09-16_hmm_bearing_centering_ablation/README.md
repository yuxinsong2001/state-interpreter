# HMM 的 bearing-specific early-reference centering 对照实验

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：run + validate
- Origin Date：2026-09-16
- Verification Status：ANALYZED
- Version Label：hmm_bearing_centering_ablation_v1

## 研究问题

第一版 HMM 直接在不同 bearing 的原始 `z∈R^8` 上拟合。Bearing1_4 在公共评分区间内全部落入 state 1。原始信号与 embedding 审计显示 Bearing1_4 实际存在持续变化，因此本实验检验：

> 跨 bearing 的初始 latent 位置偏移，是否是 HMM 状态塌缩的重要原因？

## 单因素对照

对照组使用原始 embedding：

```text
raw_z(t) = z(t)
```

实验组仅减去每个 bearing 前 15 个 embedding 的均值：

```text
centered_z(t) = z(t) - mean(z(1:15))
```

保持不变的因素：

- 相同的存档 AutoEncoder embedding；
- 相同的训练 bearing：Bearing1_1–Bearing1_3；
- 相同的三状态左到右对角高斯 HMM；
- 相同的 EM 参数和 log-domain 因果过滤；
- 相同的 step 15 起始评分区间。

本实验不做 bearing-specific 尺度归一化，避免同时改变位置和尺度。

## 结果

| Bearing | Split | raw z ρ | centered z ρ | Δρ | raw阶段切换 | centered阶段切换 |
|---|---|---:|---:|---:|---:|---:|
| Bearing1_1 | train | 0.854 | 0.916 | +0.062 | 1 | 8 |
| Bearing1_2 | train | 0.834 | 0.831 | −0.003 | 2 | 2 |
| Bearing1_3 | train | 0.914 | 0.946 | +0.032 | 2 | 2 |
| Bearing1_4 | validation | 无法计算 | 0.962 | — | 0 | 2 |
| Bearing1_5 | old holdout | 0.816 | 0.896 | +0.080 | 1 | 1 |

raw 分支与此前 log-domain HMM 逐 bearing 的 ρ、阶段切换次数和平均置信度完全一致，证明新脚本正确复现了旧对照。

### Bearing1_4

在公共评分区间中：

- raw z：状态计数为 `[0, 107, 0]`，全部为 state 1；
- centered z：状态计数为 `[70, 36, 1]`；
- centered z 在 step 85 发生 `0→1`，在 step 121 发生 `1→2`；
- centered expected stage 与 normalized lifetime 的 Spearman `ρ=0.962`。

因此，去除每个 bearing 自身的初始位置后，Bearing1_4 的状态塌缩消失。这支持“跨 bearing 初始 latent 位置偏移是原塌缩的重要原因”，但不能证明它是唯一原因。

### 其他 bearing

Bearing1_5 的 ρ 从 0.816 提高到 0.896，且仍只有一次阶段切换。Bearing1_1、1_3 提高，Bearing1_2 基本不变。

但 Bearing1_1 出现 8 次离散阶段切换，其中 step 35–47 存在三组 `0→1→0`。左到右转移限制的是隐藏状态路径，在线 filtered posterior 的逐点 argmax 仍可能在边界附近反复改变。因此，中心化修复了跨 bearing 对齐，却暴露出在线阶段输出的边界抖动问题。

## 结论

本次对照支持以下结论：

1. Bearing1_4 并非因为 embedding 没有变化而输出恒定状态。
2. 原始 z 的跨 bearing 初始位置偏移，是 HMM 发射映射失配的重要因素。
3. bearing-specific early-reference centering 是有证据支持的预处理步骤；它在 Bearing1_4 上消除了状态塌缩，并改善旧 holdout Bearing1_5。
4. 当前 centered HMM 仍不能作为最终解释器，因为 Bearing1_1 存在在线阶段抖动，置信度仍接近 1，并且状态语义没有物理标签验证。
5. 这些结果不能证明健康损伤、RUL 或维护决策有效性；normalized lifetime 仍只是时间代理。

## 下一步

保持 centered z、数据划分和 HMM 参数不变，单独比较两种在线输出规则：

- 当前规则：每一步直接取 filtered posterior 的 argmax；
- 稳定规则：加入最小持续时间或迟滞，仅改变状态读出，不重新拟合发射模型。

先验证是否能消除 Bearing1_1 的 `0↔1` 边界抖动，同时保留 Bearing1_4 的 `0→1→2`。如果不能，再考虑 GRU；不应同时加入尺度归一化、修改 HMM 和引入 GRU。

## 统计解释边界

- 所有 bearing 均已被查看，本实验是探索性消融，不是盲测。
- Bearing1_1–1_3 是 HMM 训练序列，不能作为泛化证据。
- Bearing1_4 是旧验证集，Bearing1_5 是已查看的旧 holdout，均不能恢复盲测资格。
- 第一阶段 15 点被假设为局部参考，但没有独立物理健康标签。
- 五个 bearing 的样本量不足以支持一般性方法优越结论。

## Fallacy Scan（11/11）

| 检查项 | 状态 | 说明 |
|---|---|---|
| Simpson's paradox | 未发现 | 逐 bearing 报告，没有用总体均值掩盖差异 |
| Ecological fallacy | 未发现 | 没有由工况平均推断单个 bearing |
| Berkson's paradox | 注意 | 只使用一个公开数据集和一个工况 |
| Collider bias | 不适用 | 未加入控制变量 |
| Base-rate neglect | 不适用 | 不是故障分类实验 |
| Regression to the mean | 注意 | Bearing1_4 因旧实验失败而被重点分析 |
| Survivorship bias | 无法确认 | 数据集是否包含所有失败 run 未由本实验验证 |
| Look-elsewhere effect | 注意 | 属于已见数据上的探索性消融 |
| Garden of forking paths | 注意 | 后续方案选择受到当前结果影响，必须与验证性结论分开 |
| Correlation ≠ causation | 注意 | 时间相关性不能证明物理退化因果关系 |
| Reverse causality | 注意 | normalized lifetime 不是独立损伤测量 |

## 可复现性检查

- raw 分支精确复现旧 log-domain HMM 的逐 bearing 指标；
- 新输出目录与旧配置、旧结果分离；
- 输入 latent CSV SHA-256：`1fbff8151767b124b68528ea5228f6e76cc391cc9b1ead333080d68e01fb9ec0`；
- 全量测试：`75 passed`；
- 实验执行成功，无自动重试。

## 工件

- 配置：`configs/xjtu_hmm_bearing_centering_v1.json`
- 脚本：`scripts/compare_hmm_bearing_centering.py`
- 结果：`results/2026-09-16_hmm_bearing_centering_ablation/`

