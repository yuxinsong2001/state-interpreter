# XJTU-SY 数据集适用性评估

> 评估日期：2026-08-01  
> 结论：推荐作为小型 AutoEncoder → 低维 `z` → State Interpreter 的第一阶段数据集；不能单独完成 RL state sufficiency 验证。

## 1. 数据集事实

- 15 个滚动轴承的完整 run-to-failure 加速寿命实验数据；
- 3 种固定工况，每种工况 5 个轴承：
  - 2100 rpm、12 kN；
  - 2250 rpm、11 kN；
  - 2400 rpm、10 kN；
- 两个振动通道：水平和垂直方向；
- 采样频率 25.6 kHz；
- 每分钟采集一次，每次记录 1.28 s，即 32768 个采样点；
- 每次测量保存为 CSV，第一列为水平振动，第二列为垂直振动；
- 提供最终失效部件/故障类型，但不提供每一分钟的真实连续损伤值。

主要来源：

- 官方仓库：https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets
- 数据集作者页面：https://biaowang.tech/xjtu-sy-bearing-datasets/
- 关联论文：B. Wang et al., “A Hybrid Prognostics Approach for Estimating Remaining Useful Life of Rolling Element Bearings,” IEEE Transactions on Reliability, 2020, DOI: 10.1109/TR.2018.2882682.

## 2. 与当前任务的匹配程度

| 当前研究要求 | XJTU-SY 是否支持 | 说明 |
|---|---:|---|
| 小型 AutoEncoder 输入 | 是 | 每次测量可转换成时频图或使用一维波形 |
| 低维 `z=8/16` | 是 | 可由自建 encoder 输出 |
| 同一设备的时间轨迹 | 是 | CSV 按一分钟间隔形成 run-to-failure 序列 |
| PCA/t-SNE latent trajectory | 是 | 可按时间为 embedding 着色并连接 |
| 相邻点位移 `Δ_t` | 有条件支持 | 表示相邻一分钟测量快照之间的变化，不是连续一分钟振动过程 |
| 到健康区域的距离 | 有条件支持 | 需人为定义早期健康参考窗口，并做敏感性分析 |
| 健康/退化阶段监督标签 | 不直接支持 | 数据没有逐时刻的标准阶段标签 |
| RUL/寿命进度代理 | 支持 | 可由测量序号和最终失效时间构造 normalized lifetime，但它不等于真实 damage |
| 工况泛化 | 支持 | 有三种工况，但每种仅五个轴承 |
| RL action/reward/transition | 不支持 | 无维护 action、reward 和 action-conditioned transition |

## 3. 推荐用途

XJTU-SY 适合回答：

> 一个简单 AutoEncoder 学到的低维 latent space，是否包含随轴承退化变化且跨轴承可复现的结构？

适合验证：

- latent trajectory；
- 到健康参考中心的距离；
- 相邻测量 embedding 位移；
- 连续 health indicator；
- 同工况跨轴承稳定性；
- 后续的跨工况鲁棒性。

不适合直接回答：

- 该 state 是否足以预测维护 action 的结果；
- 该 state 是否满足 RL 的 Markov 性；
- 使用该 state 是否提高维护策略 reward。

因此，它是当前第一阶段的 representation/interpretability 验证集，而不是整个 State Interpreter + RL 项目的最终数据来源。

## 4. 推荐实验顺序

### 4.1 管线调试

- 先选同一工况中的一个 bearing；
- 读取连续 CSV，确认文件编号就是时间顺序；
- 将每次测量转换为一个 `32×32` log-STFT 表示；
- 跑通 AutoEncoder、embedding 导出和曲线绘制。

单 bearing 结果只能用于调试和探索，不能作为泛化证据。

### 4.2 正式的同工况验证

先在一种工况内做 leave-one-bearing-out：每次用 4 个 bearing 建模、1 个 bearing 测试，轮换 5 次。归一化参数、AutoEncoder 训练和健康中心都只能使用当前 fold 的训练 bearings。

这样比固定一次 3/1/1 划分更适合每种工况只有 5 个 bearing 的小样本情况。

### 4.3 跨工况验证

在同工况结果成立后，再测试：

- 一个工况训练、另一个工况测试；
- 工况信息是否主导 PCA/latent 分布；
- 归一化或 condition-aware interpreter 是否必要。

## 5. 第一版预处理建议

- 第一版同时保留水平和垂直两个通道；
- 每个一分钟测量文件作为一个时间点；
- 可生成双通道 `32×32` log-STFT，或先分别生成后堆叠；
- 不要把同一个 CSV 切出的多个窗口随机分配到训练和测试；
- 如果一个 CSV 产生多个子窗口，评价和划分仍须以 bearing/measurement 为组；
- 归一化统计量只从训练 bearings 得到；
- 所有输出必须保留 `condition_id、bearing_id、measurement_index`。

## 6. 健康参考与评价边界

由于没有逐时刻 health stage ground truth，建议把训练 bearing 的最早 10% 测量暂定义为健康参考，并对 5%、10%、20% 三种定义做敏感性分析。

可以定义：

```text
μ_H = mean(z of early healthy-reference measurements in training bearings)
HI_t = ||z_t - μ_H||₂
Δ_t = ||z_t - z_(t-1)||₂
normalized_lifetime_t = t / T_failure
```

注意：

- `normalized_lifetime` 只是时间代理，不是真实 damage；
- `Δ_t` 是相邻一分钟采样快照的差异，不是连续动力学速度；
- 最终 fault element 是 run-level 信息，不能被当作每个时间点的阶段标签；
- t-SNE 只用于展示，不应在 t-SNE 坐标上计算 HI 或距离；
- reconstruction loss 低不能证明 latent 已经具有健康语义。

## 7. 最终建议

采用 XJTU-SY，并将此前工作方案中的首选数据集从 FEMTO-ST/PRONOSTIA 调整为 XJTU-SY。第一轮只做一种工况，以同工况 leave-one-bearing-out 检验 `z=8/16`、健康距离与相邻位移；结果成立后再做跨工况实验，最后才迁移到 VibFM 和 RL 环境。

