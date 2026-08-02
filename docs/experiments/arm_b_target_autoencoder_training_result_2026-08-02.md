## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-02
- Verification Status: UNVERIFIED
- Version Label: exp_result_v1

# 分支B：目标工况AutoEncoder训练结果

## Experiment Result

- **ID**：`arm_b_target_autoencoder_training_20260802`
- **类型**：training
- **状态**：completed
- **工况**：`37.5Hz11kN`
- **训练bearing**：`Bearing2_1–Bearing2_3`
- **验证bearing**：`Bearing2_4`
- **严格盲测**：`Bearing2_5`，未读取
- **运行时间**：75.5秒（训练进程内部计时）
- **退出码**：0

本步骤由 Codex 在用户授权下实现、测试和运行，不表示用户已经亲自完成或完全理解全部代码。

## 固定训练设置

```text
AutoEncoder: SmallConvAutoEncoder
latent dimension: 8
epochs: 5
batch size: 32
learning rate: 0.001
seed: 20260801
```

STFT结构与v1相同。目标工况归一化只在三个训练bearing上拟合：

```text
mean = [2.090652, 2.119821]
std  = [0.832104, 0.880762]
```

## 数据范围

| Split | Bearings | 样本数 |
|---|---|---:|
| Train | Bearing2_1、Bearing2_2、Bearing2_3 | 1,185 |
| Validation | Bearing2_4 | 42 |
| Blind holdout | Bearing2_5 | 0（未读取） |

## 训练曲线

| Epoch | Train MSE | Validation MSE |
|---:|---:|---:|
| 1 | 0.791 | 1.018 |
| 2 | 0.388 | 0.355 |
| 3 | 0.124 | 0.307 |
| 4 | 0.083 | **0.285** |
| 5 | 0.069 | 0.289 |

验证误差在epoch 4最低，因此保存epoch 4的权重。epoch 5训练误差继续下降、验证误差轻微回升，说明继续训练可能开始增加过拟合风险；按照预注册的5轮设置结束，没有增加epoch或重新调参。

checkpoint SHA-256：

```text
479f66d07916fc43556da3b74a4eae35c42f58424ff1307b76a8dafcca4f818c
```

## 可以得出的结论

- 新工况AutoEncoder训练流程已跑通；
- 模型能够明显降低训练与验证重建误差；
- 使用 `Bearing2_4` 选择出了epoch 4 checkpoint；
- 归一化、训练和validation均未使用 `Bearing2_5`。

## 不能得出的结论

- 不能仅凭重建误差判断latent适合表示健康状态；
- 不能直接把本次MSE与分支A的重建MSE比较，因为两个分支使用不同归一化统计；
- 不能证明分支B优于分支A；
- 不能修改预注册训练设置后挑选更好结果；
- 不能开始最终盲测。

## 输出工件

- `best_checkpoint.pt`：epoch 4模型、目标工况归一化、数据划分和配置哈希；
- `training_curve.csv`：5轮训练/验证MSE；
- `training_curve.png`：训练曲线；
- `training_report.json`：训练设置、样本范围、结果和checkpoint哈希。

## 下一步

使用这个冻结checkpoint，在 `Bearing2_1–Bearing2_4` 上提取有序latent并运行固定15/10 State Interpreter，得到分支B开发指标。该步骤仍不读取 `Bearing2_5`。

