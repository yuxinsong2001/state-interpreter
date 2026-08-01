# AutoEncoder State Interpreter 工作执行方案

> 制定日期：2026-08-01  
> 当前状态：实验计划已确定，尚未开始数据处理、模型训练或结果分析。

## 1. 当前任务的准确边界

本阶段不使用 VibFM checkpoint，也不比较不同 encoder 的优劣。当前问题是：

> 在一个容易控制的小型 AutoEncoder 中，能否从低维 latent representation `z` 中构造出可解释的设备健康状态？

工作主线为：

```text
按时间排列的振动测量
→ 统一预处理
→ 小型 AutoEncoder
→ z ∈ R^8 或 R^16
→ latent-space 分析
→ 简单 State Interpreter
→ 验证方法后迁移到 VibFM
```

## 2. 第一轮实验只回答三个问题

1. `z` 是否随设备退化呈现有序的空间轨迹？
2. 样本到“健康参考区域”的距离是否随退化总体增大？
3. 相邻时间点位移 `d_t = ||z_t-z_(t-1)||₂` 在健康期和退化期是否不同？

这三点目前都是待检验假设，不能提前写成结论。

## 3. 第一版的推荐实验配置

为了与老师板书对齐，并减少无关变量，第一版采用以下默认方案：

- 数据：一个公开的 run-to-failure 轴承数据集，优先 FEMTO-ST/PRONOSTIA；先使用一个 bearing 跑通流程，再扩展到多个 bearing。
- 样本顺序：必须保留同一 bearing 内的真实时间顺序。
- 输入：每个振动窗口转换为 `32×32` log-STFT 图。
- 模型：小型 2D Convolutional AutoEncoder。
- latent dimension：只比较 `8` 与 `16`。
- 训练目标：重建输入，不在第一轮加入分类、GRU、Attention 或 RL loss。
- 数据划分：按 bearing/run 划分训练集与测试集，禁止把同一 bearing 的随机窗口分散到训练集和测试集。
- 健康参考：只用训练 bearing 的早期健康窗口计算健康中心，不用测试数据拟合参考中心。

上述配置是第一版实验决策，不是老师已经固定的最终结构。

## 4. 分阶段执行流程

### 阶段 A：数据审计

需要回答：

- 数据存放在哪里，格式是什么？
- 每个文件对应设备、run、时间点还是单个窗口？
- 采样率、通道数、窗口长度和测量间隔是多少？
- 是否有 failure time、health stage 或 damage 标签？
- 能否恢复每台设备的严格时间顺序？
- 哪些 bearing 用于训练、验证和测试？

产出：`data_audit.md` 和一张数据划分表。

完成标准：能够明确构造 `(bearing_id, time_index, vibration, optional_label)`，且不存在跨设备混序。

### 阶段 B：预处理与小样本检查

步骤：

1. 读取一个 bearing 的少量连续测量。
2. 固定窗口长度和重叠率。
3. 计算 log-STFT，并调整为 `32×32`。
4. 只用训练数据统计归一化参数。
5. 绘制健康早期、中期和失效前样本，人工检查图像是否合理。

产出：预处理配置、三类示例图、shape 检查结果。

完成标准：同一配置可重复地产生有限值的 `[N, C, 32, 32]` 张量。

### 阶段 C：训练 AutoEncoder baseline

步骤：

1. 实现对称的轻量 ConvAE。
2. 分别训练 `z_dim=8` 和 `z_dim=16`。
3. 保存模型、配置、随机种子和训练曲线。
4. 比较训练/验证 reconstruction loss，并查看真实重建图。

产出：两个 checkpoint、loss 曲线、输入/重建对比图。

完成标准：模型能在未见 bearing 上产生合理重建；训练和验证 loss 没有明显发散。重建效果只证明编码器学到了信号结构，不能单独证明 `z` 表示健康。

### 阶段 D：按时间导出 latent z

每个 embedding 至少保存：

```text
bearing_id, time_index, normalized_lifetime, z_1 ... z_d, optional_label
```

产出：按 bearing 分组且按时间排序的 latent 表。

完成标准：每条 `z` 能追溯到原始测量，顺序检查通过，无数据缺失或维度错误。

### 阶段 E：建立第一版 State Interpreter

先做三个不需要复杂网络的解释器：

1. **可视化解释器**：PCA 为主，t-SNE 只作为辅助图，按时间给点着色并连接轨迹。
2. **健康参考距离**：训练健康窗口中心为 `μ_H`，定义 `HI_t = ||z_t-μ_H||₂`；之后可再测试 Mahalanobis distance。
3. **局部移动量**：`Δ_t = ||z_t-z_(t-1)||₂`，观察状态变化速度，而不是把它直接等同于损伤程度。

暂不使用 t-SNE 坐标计算距离，因为 t-SNE 会扭曲全局距离。

产出：每个 bearing 的轨迹图、`HI_t` 曲线和 `Δ_t` 曲线。

### 阶段 F：评价与模型选择

第一轮至少检查：

- reconstruction loss：编码器基本质量；
- Spearman correlation：`HI_t` 与 normalized lifetime 的秩相关；
- monotonicity：HI 是否总体单调；
- trend consistency：不同 bearing 是否呈现相似趋势；
- healthy/degraded 的 `Δ_t` 分布差异及 effect size；
- 对工况、bearing identity 和噪声的敏感性。

`z_dim=8` 与 `16` 的选择不能只看 reconstruction loss，而要优先看健康趋势是否清楚、跨 bearing 是否稳定、结果是否容易解释。

## 5. 何时升级到其他 Interpreter

- 如果单窗口 `z` 已产生稳定健康趋势：保留距离/HI 方法作为第一版，不必增加 GRU。
- 如果轨迹噪声较大但短期历史能明显改善趋势：再测试平滑、HMM、GRU 或 TCN。
- 如果存在清楚且稳定的阶段簇：再测试 clustering 或阶段分类。
- 只有 Attention 相比简单时间模型带来可重复增益时才保留 Attention。
- 只有 representation 层验证通过后，才进入 RL state sufficiency、reward/transition prediction 和后续 VibFM 迁移。

## 6. 立即开始的任务

当前只做阶段 A，不训练模型。

1. 确认本地是否已有 FEMTO-ST/PRONOSTIA 或其他连续 run-to-failure 数据。
2. 如果已有，定位数据根目录并查看一个 bearing 的文件结构。
3. 填写 `data_audit.md`：数据集、bearing 数量、文件格式、采样率、通道、测量顺序、标签和建议划分。
4. 在数据审计完成后，再固定 STFT 参数和 AutoEncoder 输入接口。

当前检查点：**在数据位置、时间顺序和划分单位没有确认前，不编写训练脚本。**

## 7. 第一轮应避免的错误

- 随机按窗口划分训练和测试，造成同一 bearing 的信息泄漏。
- 用全部数据（包括测试 bearing）计算归一化参数或健康中心。
- 把 t-SNE 图上的视觉分离直接当作定量证据。
- 把 reconstruction loss 低误认为健康状态表示已经有效。
- 同时改变数据、模型、latent 维度和 interpreter，导致无法判断改进来源。
- 提前加入 GRU、Attention 或 RL，使第一轮结果无法解释。

## 8. 下一次组会可汇报的最小成果

- 一张端到端研究流程图；
- 数据审计与无泄漏的数据划分方案；
- AutoEncoder 结构和 `z=8/16` 的实验设计；
- 三个待验证的 State Interpreter 指标；
- 如果阶段 C–E 已完成，再展示重建图、latent trajectory、HI 与 `Δ_t` 曲线；
- 明确哪些是实验结果，哪些仍是研究假设。

