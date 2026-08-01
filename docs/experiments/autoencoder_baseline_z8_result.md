## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-01
- Verification Status: VERIFIED
- Version Label: ae_baseline_z8_v1

# AutoEncoder z=8 首轮短训练结果

## 实验目的

验证完整工程管线是否能够在不发生bearing级数据泄漏的条件下完成：

```text
XJTU-SY CSV
→ log-STFT [2,32,32]
→ train-only channel normalization
→ SmallConvAutoEncoder
→ z ∈ R^8
→ reconstruction
```

本次不是正式模型比较，也不评价latent space的健康语义。

## 数据划分

仅使用 `35Hz12kN` 工况：

- 训练：`Bearing1_1`、`Bearing1_2`、`Bearing1_3`，共442个测量；
- 验证：`Bearing1_4`，共122个测量；
- 留出测试：`Bearing1_5`，共52个测量，本次未读取和未评价。

归一化均值和标准差只使用训练bearings拟合：

| 通道 | mean | std |
|---|---:|---:|
| horizontal | 2.575276 | 0.697269 |
| vertical | 2.373167 | 0.771111 |

## 训练配置

- latent dimension：8；
- epochs：5；
- batch size：32；
- optimizer：Adam；
- learning rate：0.001；
- loss：MSE reconstruction loss；
- seed：20260801；
- device：CPU；
- checkpoint selection：最低validation loss。

## 训练结果

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| 1 | 0.936011 | 0.856709 |
| 2 | 0.751976 | **0.615747** |
| 3 | 0.664923 | 0.675687 |
| 4 | 0.592651 | 0.663113 |
| 5 | 0.369078 | 0.670691 |

最佳checkpoint为epoch 2。训练loss在之后继续下降，而validation loss没有继续改善，提示首轮设置存在早期过拟合倾向。

## 软件验证

- checkpoint可重新加载；
- encoder输出shape为 `[1,8]`；
- reconstruction输出shape为 `[1,2,32,32]`；
- 输出全部有限；
- 仓库21项单元测试通过；
- 留出测试bearing未评价。

## 输出文件

- `state-interpreter/runs/ae_baseline_z8_20260801/best_checkpoint.pt`
- `state-interpreter/runs/ae_baseline_z8_20260801/training_curve.csv`
- `state-interpreter/runs/ae_baseline_z8_20260801/training_report.json`

## 结论边界

该实验只证明数据划分、训练集专用归一化、DataLoader、训练循环、checkpoint选择和8维输出能够端到端运行。重建loss下降不能证明 `z` 已经表示健康状态，也不能据此选择 `z=8` 优于 `z=16`。

## 下一步

1. 生成验证bearing的输入—重建对比图；
2. 在完全相同的数据划分与训练配置下训练 `z=16`；
3. 比较验证重建误差，但不只凭重建误差选择latent维度；
4. 导出按时间排序的 `z`，开始PCA trajectory、健康中心距离和相邻位移分析；
5. 在方法固定前继续保持 `Bearing1_5` 不可见。
