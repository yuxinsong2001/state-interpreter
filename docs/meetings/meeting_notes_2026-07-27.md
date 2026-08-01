# 2026-07-27 最新组会纪要

## 一、老师明确的新方向

老师对当前项目路线进行了重要调整：

> 现阶段不直接使用 VibFM，也不需要申请 VibFM checkpoint。首先自己建立一个小型 AutoEncoder，通过其 Encoder 得到低维 latent embedding `z`，在这个简化空间中研究 State Interpreter。待方法和评价方式明确后，再迁移到 VibFM。

新的阶段性主线是：

```text
输入信号
    ↓
小型 AutoEncoder
    ↓
只保留 Encoder
    ↓
低维 latent embedding z（例如 8 或 16 维）
    ↓
State Interpreter
    ↓
可解释健康状态、轨迹或曲线
```

后续路线才是：

```text
在小型 AutoEncoder 上验证 State Interpreter
    ↓
理解哪些解释方式有效
    ↓
把已验证的方法迁移到 VibFM embedding
```

## 二、为什么暂时不使用 VibFM

老师给出的理由是：

- VibFM 是大型模型；
- VibFM embedding 维度较高，例如可能达到 256 维；
- 高维空间较难直接判断 State Interpreter 是否真正有效；
- 使用 checkpoint 和大模型会增加计算与调试成本；
- 当前研究重点不是追求最强 feature extractor，而是理解如何利用 embedding 表示状态。

因此，第一阶段希望得到：

```text
z ∈ R^8
```

或：

```text
z ∈ R^16
```

而不是立即处理：

```text
z ∈ R^256
```

小型 AutoEncoder 的目的不是替代 VibFM 作为最终模型，而是建立一个容易观察、容易调试的研究平台。

## 二点一、板书对新路线的进一步确认

老师的板书可以分成左、中、右三个区域。

### 左侧：小型 AutoEncoder / Encoder

左侧画出了：

```text
AE
→ 32×32
→ 16×16
→ 8×8
→ 4×4
→ latent z
```

这表示通过 Encoder 逐步压缩输入。板书中的空间尺寸更像是压缩过程示意，不能直接理解为已经确定的卷积层配置。

在压缩末端附近还写有类似：

```text
z^256
```

结合组会语音，它应理解为高维 latent representation 的参考或当前复杂情况，而不是本轮实验必须采用的维度。板书左下方用红框明确写出了：

```text
z^8 / z^16
```

这与老师口头说明一致：第一阶段希望自己构造 8 维或 16 维的低维 latent representation。

### 中间红框：当前真正需要研究的 State Interpreter

红色大框中包含多个问号，说明这里的具体方法尚未确定，需要通过研究和实验回答。

板书明确给出两条候选分析思路。

#### 思路 1：t-SNE / latent trajectory

上方图中画出了一条从低值逐渐上升并接近 1 的曲线，并标注 `t-SNE`。这表明老师希望观察：

- 样本在低维投影空间中的位置；
- 健康到损坏的轨迹；
- 是否可以把轨迹或位置转成 0–1 的健康曲线；
- latent space 是否包含连续状态结构。

这里需要注意：t-SNE 通常是二维可视化方法，图中的 0–1 曲线更可能表达“由 latent structure 得到的候选健康状态”，而不是说 t-SNE 本身直接输出可靠 HI。

#### 思路 2：相邻点移动距离

下方图中标出了：

```text
Δp
Healthy
Kaputt
```

其中 Healthy 对应较小移动，Kaputt 对应较大移动。可以形式化为：

```text
Δz_t = z_t - z_(t-1)
```

以及：

```text
d_t = ||z_t - z_(t-1)||
```

板书要表达的候选假设是：

> 健康状态下 latent point 的移动可能较小，而故障或退化阶段的移动可能更大。

这仍然是假设，需要用连续时间数据验证，不能预先当作事实。

### 右侧：后期迁移到 VibFM

红框右侧画出了指向 `VibFM` 的箭头，结合老师口头说明，应理解为：

```text
先在低维 AutoEncoder latent space 中研究解释方法
→ 方法可行后
→ 再应用于 VibFM
```

因此，VibFM 没有从项目中取消，而是被放到了后续迁移阶段。

### 板书的整体含义

```text
小型 AE / Encoder
→ z∈R^8 或 R^16
→ [待研究区域]
   ├── t-SNE/PCA 与状态轨迹
   ├── 相邻 latent 位移 Δz
   ├── 健康到损坏的位置变化
   └── 可解释曲线或状态
→ 后期迁移到 VibFM
```

## 三、AutoEncoder 在项目中的作用

老师强调，小型 AutoEncoder 的主要任务是压缩输入。

训练时：

```text
x
→ Encoder
→ z
→ Decoder
→ x_hat
```

使用重建目标：

```text
x_hat ≈ x
```

AutoEncoder 训练完成后，主要使用 Encoder：

```text
z = Encoder(x)
```

老师举例说明，输入表示可以逐级压缩：

```text
32 × 32
→ 16 × 16
→ 8 × 8
→ 4 × 4
```

该例子用于说明压缩过程，并不代表最终网络结构已经确定。

## 四、State Interpreter 的任务仍然是什么

State Interpreter 仍然位于 Encoder 之后：

```text
输入
→ Encoder
→ z
→ State Interpreter
→ 可解释输出
```

老师给出的核心问题是：

> 如何使用 `z`，使其最终产生具有健康含义的输出？

可能的解释方式包括：

### 1. Latent-space 可视化

使用：

- t-SNE；
- PCA；
- 其他二维或三维投影。

目的：

- 观察健康和损坏样本是否形成不同区域；
- 观察退化样本在 latent space 中如何移动；
- 检查 embedding 是否包含状态结构。

### 2. Latent-space 距离

研究相邻时间点在 latent space 中的运动：

```text
Δz_t = z_t - z_(t-1)
```

或距离：

```text
d_t = ||z_t - z_(t-1)||
```

可以进一步研究：

- 相邻状态移动了多远；
- 是否出现明显变点；
- 退化时移动速度是否增加；
- 到健康区域的距离是否可以作为 Health Indicator。

### 3. Latent-space 位置

老师举例：

```text
位置 1 → 健康
位置 4 → 损坏
```

也就是说，可以训练或构造一个任务，让模型判断样本在 latent space 中所处的状态区域。

候选输出包括：

- Healthy / Damaged；
- 多个健康阶段；
- 一条连续健康曲线；
- latent trajectory；
- 到健康参考区域的距离。

## 五、老师目前并未确定的内容

以下内容仍然是开放问题：

- AutoEncoder 的具体输入是原始振动、STFT 还是其他特征；
- latent dimension 最终选择 8 还是 16；
- State Interpreter 最终输出连续 HI、离散阶段，还是两者；
- 使用什么数据集；
- 使用什么标签或辅助训练任务；
- 第一版是否需要时序模型；
- 如何评价 latent space 的状态表达能力。

因此，当前需要通过小规模实验比较，而不是假设最终结构已经确定。

## 六、对原工作计划的修正

### 原先计划

```text
获得 VibFM checkpoint
→ 提取真实 z_health
→ 建立 State Interpreter
```

### 最新计划

```text
准备简单数据
→ 自己训练小型 AutoEncoder
→ 提取低维 z
→ 分析 latent space
→ 建立简单 State Interpreter
→ 验证解释方法
→ 后期迁移到 VibFM
```

因此：

- 暂时不需要向老师索取 VibFM checkpoint；
- 暂时不实现真实 `VibFMAdapter`；
- 现有独立 State Interpreter 仓库仍然保留；
- 需要为 AutoEncoder 增加独立模块或 adapter；
- 之前整理的 HI、cluster、HMM 和 predictive-state 文献仍然有价值，但实验顺序需要提前加入 AutoEncoder latent-space baseline。

## 七、现在最直接的任务

### 任务 1：继续文献检索

重点检索：

- AutoEncoder latent space for bearing health;
- interpretable latent representation for PHM;
- latent trajectory degradation;
- latent-space distance health indicator;
- low-dimensional state representation;
- t-SNE/PCA visualization of bearing degradation;
- latent dimension selection for AutoEncoder.

阅读时需要记录：

- 输入表示；
- AutoEncoder 结构；
- latent dimension；
- 重建损失；
- State/HI 的定义；
- latent space 如何解释；
- 是否使用时间顺序；
- 评价方法；
- 是否有公开代码。

### 任务 2：建立小型 AutoEncoder baseline

第一版建议保持简单：

```text
输入
→ Encoder
→ z（8 或 16 维）
→ Decoder
→ 重建输入
```

需要至少比较：

- latent dimension 8；
- latent dimension 16。

### 任务 3：输出低维 z

训练完成后，按：

```text
unit/run/time
```

保存：

- sample ID；
- time index；
- health/damage label（如果有）；
- latent vector `z`；
- reconstruction error。

### 任务 4：分析 latent space

至少完成：

- PCA；
- t-SNE；
- latent trajectory；
- 相邻距离 `||z_t-z_(t-1)||`；
- 到健康中心的距离；
- reconstruction error 随时间变化。

### 任务 5：尝试简单 State Interpreter

首先考虑：

```text
z
→ continuous HI
```

以及：

```text
z
→ Healthy / Degraded / Damaged
```

暂时不需要直接使用 GRU + Attention。

## 八、建议下次组会展示

如果数据能够及时准备，下次组会可以展示：

1. 小型 AutoEncoder 结构；
2. latent dimension 8 与 16 的重建结果；
3. reconstruction loss；
4. PCA/t-SNE latent-space 图；
5. 一个样本序列在 latent space 中的轨迹；
6. 相邻 latent distance 或健康中心距离曲线；
7. 初步判断这种 `z` 是否能够表达健康状态。

如果尚未完成训练，也至少应展示：

1. 文献中低维 latent state 的实现矩阵；
2. 确定的数据输入和数据集；
3. AutoEncoder 实验设计；
4. latent-space 评价方案。

## 九、需要保留的结论边界

- t-SNE 图只能说明可视化结构，不能单独证明状态可用于 RL；
- reconstruction error 较低不代表 latent space 一定包含健康信息；
- Healthy/Damaged 可分不代表状态已经 decision-ready；
- 当前简化 AutoEncoder 是研究工具，不是最终替代 VibFM；
- 迁移到 VibFM 是后续阶段，而不是被取消。

## 十、一句话总结

> 最新任务是先自行训练一个低维小型 AutoEncoder，用 8 或 16 维 latent embedding 研究状态可视化、距离、轨迹和健康解释方法；在简化模型上理解 State Interpreter 如何工作后，再将方法迁移到 VibFM。
