## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-02
- Verification Status: UNVERIFIED
- Version Label: code_plan_v2.1

# State Interpreter v2.1 双分支跨工况实验修订

## 修订原因

初版 v2 只预注册了“在新工况重新训练 AutoEncoder，再使用固定 State Interpreter”的分支。这可以检验解释方法迁移，但没有回答另一个重要问题：由 `35Hz12kN` 训练的完整系统能否不经适配直接泛化到 `37.5Hz11kN`。

在尚未查看 `Bearing2_5` 的信号统计、latent、重建误差或 state 输出之前，本修订增加直接泛化分支。初版文件和配置保留作为研究决策记录；实际执行以 v2.1 为准。

## 共同数据边界

```text
target condition: 37.5Hz11kN
development/train: Bearing2_1, Bearing2_2, Bearing2_3
validation:        Bearing2_4
strict blind:      Bearing2_5
```

两分支必须共享完全相同的原始输入顺序、STFT定义、State Interpreter参数、状态定义和指标。`Bearing2_5` 只允许由最终联合评估脚本读取一次，不能分别打开两次结果。

## 分支A：直接泛化（zero-shot system transfer）

```text
37.5Hz11kN vibration
→ v1 STFT
→ v1 normalization fitted on Bearing1_1–1_3
→ frozen v1 AutoEncoder
→ frozen State Interpreter 15/10
→ [level, trend, movement]
```

固定输入工件：

- checkpoint：`runs/ae_baseline_z8_20260801/best_checkpoint.pt`
- checkpoint SHA-256：`6edb626d1961cb1b1a7a2c32c52c9c96b0100cabb712cda4d79ab3c2a2e7695a`
- v1配置 SHA-256：`6fcdcc7852fe356c5897c89edf4b3fa83baf95105af84bc4cc90043f5b72ea26`

该分支不允许重新拟合归一化、不允许更新 AutoEncoder、不允许修改 State Interpreter。它回答完整系统是否具有零样本跨工况能力。

## 分支B：Encoder适配后迁移（adapted encoder transfer）

```text
37.5Hz11kN vibration
→ same STFT
→ normalization fitted only on Bearing2_1–2_3
→ same-architecture z=8 AutoEncoder trained on Bearing2_1–2_3
→ checkpoint selected with Bearing2_4 reconstruction loss
→ frozen State Interpreter 15/10
→ [level, trend, movement]
```

State Interpreter 不参与训练或调参。该分支回答：当 Encoder 适应新工况后，相同解释规则能否迁移。

## 公平比较规则

除 Encoder 权重和归一化来源外，两分支必须保持一致：

- 相同 `Bearing2_5` CSV 和时间顺序；
- 相同 STFT 参数与 `[2,32,32]` 输出；
- 相同 `z=8`；
- 相同 `calibration_steps=15`；
- 相同 `temporal_window=10`；
- 相同 `[level, trend, movement]` 计算；
- 相同指标代码；
- 相同的盲测打开时间。

## 主要指标和联合解释

对两分支分别计算 `Bearing2_5` READY states 中 `level` 与 normalized lifetime 的 Spearman ρ。单分支解释仍为：

- `ρ ≥ 0.70`：较强支持；
- `0.30 ≤ ρ < 0.70`：有限支持；
- `ρ < 0.30` 或为负：不支持。

同时报告：

\[
\Delta\rho=\rho_{B,adapted}-\rho_{A,direct}
\]

联合结果按下表解释：

| 分支A | 分支B | 解释 |
|---|---|---|
| 成功 | 成功 | 完整系统可以直接泛化，Encoder适配不是必要条件 |
| 失败 | 成功 | Interpreter可以迁移，但Encoder/归一化需要适配新工况 |
| 失败 | 失败 | 当前表示与解释组合没有得到跨工况支持 |
| 成功 | 失败 | 优先检查新AutoEncoder训练或latent尺度；单次结果不足以证明旧Encoder更优 |

“成功”在此指 `ρ≥0.70`，只适用于预注册的案例级工程判断，不等同于统计总体结论。

## 执行顺序

1. 实现配置驱动的数据划分和泄漏保护测试。
2. 实现分支A的开发集推理，但只查看 `Bearing2_1–2_4`。
3. 训练分支B AutoEncoder，并只在 `Bearing2_1–2_4` 完成开发检查。
4. 建立一个联合评估脚本，启动前验证两个 checkpoint/config 哈希和 `blind_holdout_evaluated=false`。
5. 冻结代码和两分支输入工件。
6. 联合脚本只读取一次 `Bearing2_5`，同时产生A/B结果和差值。
7. 评估后将配置标记为已使用，禁止根据结果修改 v2.1。

## 当前门槛

本次只修订实验设计，没有执行推理或训练。下一项实现任务仍是配置驱动的数据划分接口和泄漏保护测试，但接口必须同时服务于直接泛化和适配Encoder两个分支。

