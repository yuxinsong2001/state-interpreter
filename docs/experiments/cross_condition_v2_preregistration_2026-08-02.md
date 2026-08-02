## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-02
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

# State Interpreter v2 跨工况实验预注册

## Experiment Overview

- **标题**：从 `35Hz12kN` 到 `37.5Hz11kN` 的 State Interpreter 迁移验证
- **研究问题**：固定第一版 State Interpreter 的定义与参数后，它能否在另一工况下、由同结构 AutoEncoder 产生的新 latent space 中仍输出与退化进程一致的状态？
- **假设**：个体早期校准和相对时序特征能降低 bearing 个体差异，因此 `level` 在严格留出的 `Bearing2_5` 上仍应与 normalized lifetime 呈明显正相关。
- **类型**：training + analysis
- **当前状态**：数据审计和协议锁定已完成；尚未训练、提取 latent 或查看 `Bearing2_5` 的状态结果。

## 为什么本轮不直接测试整个旧模型的零样本迁移

如果同时固定由 `35Hz12kN` 训练的 AutoEncoder，并直接处理新工况，一旦失败，将无法判断原因来自：

1. AutoEncoder/归一化不能跨工况；还是
2. State Interpreter 不能解释新 latent。

本课题当前研究对象是 State Interpreter，因此 v2 首先保持 AutoEncoder 的结构、训练规则和 `z=8` 不变，但在新工况训练一个新的 AutoEncoder；State Interpreter 本身保持不变。这样主要检验“解释规则能否迁移到新的低维表示”。整个系统的零样本跨工况能力作为后续独立实验，不能与本轮混合。

## 第二工况数据审计

数据目录：`37.5Hz11kN`，转速 37.5 Hz，径向载荷 11 kN。

| Bearing | CSV 数量 | 编号范围 | 缺号 | 总字节数 |
|---|---:|---:|---:|---:|
| Bearing2_1 | 491 | 1–491 | 0 | 658,613,792 |
| Bearing2_2 | 161 | 1–161 | 0 | 210,478,842 |
| Bearing2_3 | 533 | 1–533 | 0 | 701,247,122 |
| Bearing2_4 | 42 | 1–42 | 0 | 55,317,954 |
| Bearing2_5 | 339 | 1–339 | 0 | 440,562,684 |

合计 1,566 个 CSV。每个 bearing 的首、中、末文件共15个文件已抽查，均满足：

- 32,768 行、2列；
- 表头为 `Horizontal_vibration_signals,Vertical_vibration_signals`；
- 抽查的首行数据均可解析为有限浮点数；
- 文件编号连续。

本次只查看了原始目录结构和格式，没有计算 `Bearing2_5` 的振动统计、latent、重建误差或 State Interpreter 输出，因此不消耗其严格盲测资格。

## 固定数据划分

```text
AutoEncoder training:   Bearing2_1, Bearing2_2, Bearing2_3
AutoEncoder validation: Bearing2_4
Strict blind holdout:   Bearing2_5
```

禁止按 CSV 随机拆分；归一化统计只能由三个训练 bearings 拟合。`Bearing2_5` 在最终一次性评价前不得用于模型选择、参数调整、曲线查看或阈值制定。

## 固定部分

- STFT：Hann window，`n_fft=1024`、`win_length=1024`、`hop_length=512`、`log1p` magnitude、输出 `[2,32,32]`；
- AutoEncoder：沿用 v1 的网络结构、`z=8`、batch size 32、learning rate 0.001、5 epochs 和 seed 20260801；
- State Interpreter：`calibration_steps=15`、`temporal_window=10`；
- 状态：`[level, trend, movement]`；
- 每个 bearing 开始前调用 `reset()`，只按时间顺序在线更新；
- normalized lifetime 只作为排序一致性的代理指标，不称为真实 damage label。

## 允许变化的部分

- 新工况训练集拟合得到的通道归一化均值和标准差；
- 新工况 AutoEncoder 的模型权重；
- 由新工况训练自然产生的训练/验证重建误差。

不允许根据新工况结果修改 State Interpreter 的15/10参数或三个状态定义。若发现代码错误，应记录、修复并重新建立一个带新版本号的协议，不能覆盖本预注册。

## 分阶段执行

1. 扩展训练脚本，使 bearing split 可由命令行或配置显式传入，移除对 `Bearing1_*` 的硬编码。
2. 为 split 校验增加测试，确保训练、验证和 holdout 互斥，且归一化只拟合训练 bearings。
3. 在 `Bearing2_1–2_3` 上训练相同结构 AutoEncoder，以 `Bearing2_4` 重建误差选择 checkpoint。
4. 提取 `Bearing2_1–2_4` 的有序 latent，执行管线和描述性检查；不得调 State Interpreter 参数。
5. 在代码、配置和解释标准均冻结后，只对 `Bearing2_5` 执行一次最终评价。

## Analysis Plan

### 主要指标

`Bearing2_5` READY states 中，`level` 与 normalized lifetime 的 Spearman ρ。

预先规定解释：

- `ρ ≥ 0.70`：提供较强的跨工况支持；
- `0.30 ≤ ρ < 0.70`：只提供有限支持；
- `ρ < 0.30` 或为负：当前迁移方案不受支持。

### 次要描述指标

- directional monotonicity；
- smoothness；
- trend noise；
- READY 数量与比例；
- level 首末差；
- movement 峰值及其时间位置。

次要指标不得被单独挑选为成功证据。当前样本单位是 bearing，只有一个严格 holdout bearing，因此本轮是工程验证和案例级证据，不做总体显著性推断。

## Expected Outputs

| 输出 | 计划路径 | 成功标准 |
|---|---|---|
| v2 配置 | `configs/xjtu_cross_condition_v2.json` | JSON 可解析且划分互斥 |
| 新工况 checkpoint | `runs/ae_z8_37_5hz11kn_v2/` | 训练完成且保存验证最佳权重 |
| latent 轨迹 | `runs/latent_analysis_z8_37_5hz11kn_v2/` | bearing/step 连续且仅含预定数据 |
| blind 报告 | 后续锁定时确定 | 只包含 Bearing2_5，保存配置与输入哈希 |

## 当前结论与下一门槛

数据足以进入 v2。下一项代码任务不是立即训练，而是先把训练脚本中的 bearing 划分改成显式配置，并用测试证明 holdout 不会进入训练、归一化或 validation。

