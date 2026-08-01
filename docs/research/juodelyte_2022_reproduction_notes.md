# Juodelyte et al. (2022) 精读与复现笔记

更新日期：2026-07-25  
论文：*Predicting Bearings' Degradation Stages for Predictive Maintenance in the Pharmaceutical Industry*  
论文链接：https://arxiv.org/abs/2203.03259  
作者代码：https://github.com/DovileDo/BearingDegradationStageDetection  
本地代码：`literature_reproduction/BearingDegradationStageDetection`

## 1. 这篇论文在当前工作主线中的位置

当前项目的主问题是：

> 如何把 VibFM 的 `z_health` 转换成可解释、连续、可用于决策的设备状态？

本论文提供了其中一条可复现路线：

`振动信号 → latent representation → 自动划分退化阶段 → 监督分类器预测当前阶段`

第一轮复现先保留作者的 AutoEncoder。跑通并理解以后，再把 AutoEncoder latent 替换成 VibFM `z_health`。这样可以区分：

- 原论文的数据、标签和分类流程是否能工作；
- 替换成 VibFM 后是否产生额外收益。

## 2. 方法整体结构

作者方法分成两个部分。

### Part 1：自动生成退化阶段标签

```text
FEMTO-ST/PRONOSTIA 原始振动
        ↓
按 bearing 合并连续测量
        ↓
水平和垂直振动分别转换到频域
        ↓
AutoEncoder 学习低维 representation
        ↓
K-means 对 latent representation 聚类
        ↓
按照簇在生命周期中出现的平均时间排序
        ↓
结合 AE 重建误差阈值生成 4 个阶段
```

### Part 2：训练统一阶段分类器

```text
水平频谱 641维 ─→ Dense branch ─┐
                               │
垂直频谱 641维 ─→ Dense branch ─┼→ Fusion MLP → 4阶段概率
                               │
时域统计特征 26维 → Dense branch ┘
```

分类器输出：

- Healthy；
- Fault/Degradation Stage 1；
- Fault/Degradation Stage 2；
- Fault/Degradation Stage 3。

## 3. 从源码核验到的具体实现

### 3.1 频域预处理

来源：`src/data/TransformToFrequencyDomain.py`

- 每个处理窗口包含 2560 个采样点；
- 水平 `Hacc` 和垂直 `Vacc` 分别处理；
- 首先使用 `scipy.signal.decimate(..., 2)` 做 2 倍降采样；
- 对降采样后的 1280 点信号执行 real FFT；
- 每个方向得到 641 个幅值，因此总频域输入为 1282 维。

### 3.2 AutoEncoder

来源：`src/models/AutoEncoder.py`

单方向网络：

```text
641 → 256 → 128 → 64 → 32 → 8
                      latent
8 → 8 → 32 → 64 → 128 → 641
```

- 激活函数：隐藏层 ReLU，输出层 Linear；
- 优化器：Adam；
- 重建损失：MAE；
- 训练：100 epochs，batch size 64；
- 水平和垂直方向分别训练；
- 两个 8 维 encoding 拼接成 16 维 representation。

### 3.3 四阶段标签如何产生

这是源码中最关键的逻辑：

1. 取每个 bearing 前 80% 的数据作为 `X_truncated`；
2. 分别训练水平和垂直 AutoEncoder；
3. 对两个方向的 8 维 encoding 拼接；
4. 对 16 维 encoding 执行 `KMeans(n_clusters=3)`；
5. 根据每个簇在生命周期中出现的平均位置，将三个簇排序为较早到较晚；
6. 使用水平信号的 AE 重建误差计算异常阈值：

   `threshold = mean(training reconstruction error) + 3 × std`

7. 超过阈值的观测被映射到最严重阶段，从而最终得到四个标签。

注意：README 的简化表述容易让人误以为 K-means 直接生成四阶段。源码实际上是“三个 K-means 簇 + 一个 AE 异常阶段”。

### 3.4 26 个时域特征

来源：`src/data/TransformToTimeDomain.py`

每个方向计算 13 个特征：

- zero crossing；
- kurtosis；
- RMS；
- peak ratio；
- mean；
- standard deviation；
- median absolute value；
- skewness；
- crest factor；
- energy；
- Shapiro statistic；
- KL divergence；
- reverse KL divergence。

水平与垂直合计 26 维。

### 3.5 分类器

来源：`src/models/NNclassifier.py`

- 水平频谱分支：`641 → 256 → 128 → 64 → 32 → 4`；
- 垂直频谱分支：同上；
- 时域元特征分支：`26 → 16 → 4`；
- 拼接后：`12 → 64 → 32 → 4`；
- 输出：4 类 softmax；
- 损失：categorical cross-entropy；
- 频谱分支主要 Dense 层使用 L2 regularization。

### 3.6 评价方式

作者代码包含：

- 自动标签与人工标签的逐阶段 F1；
- 分类器相对于 AE/PCA 自动标签的逐阶段 F1；
- 5 点 moving average 后的阶段 posterior；
- 某阶段区间被其他阶段预测结果穿插的比例；
- 首次预测 Stage 2/3 后又返回 Healthy/Stage 1 的比例；
- 首次故障判断时剩余的生命周期比例。

最后两项与当前项目计划中的“逆向状态转移次数”和“告警提前量”非常接近。

## 4. 当前发现的复现风险

### 4.1 环境文件不能直接使用

作者的 `requirements.txt` 是完整旧 Conda 环境导出，包含大量与项目无关的软件和本地构建路径。关键版本包括：

- TensorFlow 1.13.1；
- TensorBoard 1.13.1；
- NumPy 1.21.4；
- 较旧的 Pandas、SciPy 和 Scikit-learn 环境。

该环境不适合直接在当前现代 Python 环境安装。应建立最小依赖并把少量旧 Keras API 更新到现代 TensorFlow/Keras，或者先只复现标签生成部分。

### 4.2 数据未包含在仓库中

仓库没有 FEMTO 数据，需要单独下载 NASA PCoE 中的 FEMTO Bearing/PRONOSTIA 数据，并按照 README 放入 `data/raw/`。

### 4.3 “前 80%”并不等于健康数据

代码把每个 bearing 前 80% 用于训练 AE。对于 run-to-failure 数据，前 80% 很可能已经含有退化样本。因此：

- 重建误差不一定是真正的“健康分布距离”；
- anomaly threshold 可能受到退化样本污染；
- 后续应增加“仅用早期健康段训练”的对照。

### 4.4 标签具有时间排序假设

K-means 簇本身没有 Healthy/Stage 1/Stage 2 的语义。代码根据簇出现的平均时间位置排序，相当于引入：

> 越晚出现的簇代表越严重的退化。

这一假设对单调 run-to-failure 数据合理，但必须明确写入方法，而不能把聚类标签当作自然具有物理含义。

### 4.5 原评价存在循环性

分类器测试时的一部分“真值”仍由同一类 AE/PCA 自动标签方法生成。这能验证分类器是否学会复现标签器，却不能独立证明阶段具有真实物理意义。

更可靠的对照应包括：

- 人工标签；
- failure endpoint；
- RMS/频谱变化点；
- 已知 damage 或寿命比例；
- 跨 bearing 泛化。

## 5. 分步复现任务

### 步骤 1：论文与代码结构分析

状态：已完成第一轮。

输出：

- 方法数据流；
- 标签生成逻辑；
- 模型结构；
- 评价指标；
- 环境和复现风险。

### 步骤 2：准备最小运行环境

待完成：

1. 确认当前 Python 与 TensorFlow/PyTorch 环境；
2. 建立独立环境；
3. 从完整 `requirements.txt` 提取最小依赖；
4. 对旧 Keras API 做最少量兼容修改；
5. 先执行无需数据的 import/model build smoke test。

建议的最小依赖：

- numpy；
- pandas；
- scipy；
- scikit-learn；
- matplotlib；
- tensorflow/keras。

### 步骤 3：准备 FEMTO 数据

待完成：

1. 下载 FEMTO Bearing/PRONOSTIA；
2. 核验 bearing ID、训练/测试划分和时间顺序；
3. 保留水平与垂直振动；
4. 运行合并脚本；
5. 检查每个 bearing 的窗口数量和是否按时间排序。

### 步骤 4：只复现自动标签器

待完成：

1. 频域特征提取；
2. AutoEncoder 训练；
3. 提取 16 维 latent；
4. K-means 三簇；
5. 重建误差补充第四阶段；
6. 绘制阶段随时间变化；
7. 检查是否发生大量阶段逆向跳转。

### 步骤 5：复现监督分类器

待完成：

1. 构造 1282 维频域特征；
2. 构造 26 维时域特征；
3. 使用自动阶段标签训练分类器；
4. 按 bearing 测试；
5. 计算逐阶段 F1、Macro F1、Balanced Accuracy 和逆向转移。

### 步骤 6：建立更严格的 baseline

待完成：

- PCA → K-means；
- AutoEncoder → K-means；
- 仅健康段 AE reconstruction error；
- One-Class SVM/SVDD；
- 不使用时序平滑与使用时序平滑的比较。

### 步骤 7：替换为 VibFM

依赖老师提供 checkpoint。

```text
原方案：
频谱 → AutoEncoder → 16维 latent → K-means

迁移方案：
STFT → frozen VibFM → z_health → 标准化/可选 PCA → K-means
```

需要公平比较：

- 相同 bearing 划分；
- 相同阶段生成方法；
- 相同分类器或相同轻量 head；
- 相同评价指标；
- AE latent 与 `z_health` 不允许使用不同的测试信息。

## 6. 当前下一步

当前立即执行的是“步骤 2：准备最小运行环境”。老师回复 checkpoint 和目标数据集后，再决定是否直接推进 VibFM 替换；在此之前，原论文复现仍可继续。

### 本机环境初检

- 系统 Python：3.13.3；
- NumPy：2.4.4；
- Pandas：3.0.2；
- SciPy：1.17.1；
- Scikit-learn：1.8.0；
- TensorFlow：未安装；
- 当前工作区没有现成 `.venv`。

结论：不能在系统 Python 中直接运行作者的 TensorFlow 1.13.1 代码。下一步应创建独立复现环境，并在“现代 TensorFlow 兼容迁移”与“使用 PyTorch 等价重写 AE”之间作出可复现选择；为了忠实复现，优先采用现代 TensorFlow/Keras 的最小兼容迁移。

## 7. 研究定位更正

VibFM 是当前项目的既定前提，不是需要通过本论文复现重新选择的候选 encoder。因此，本笔记中涉及“AE latent 与 `z_health` 公平比较”的内容只应理解为可选 sanity check，不是核心研究目标。

阅读和有限复现本论文的主要目的调整为：

1. 理解 latent representation 如何被转换为 degradation stages；
2. 学习 pseudo-label、时间排序、异常阈值和阶段分类器的设计；
3. 提取可迁移到 `z_health` 后面的 State Interpreter 模块；
4. 复用阶段 F1、逆向转移、阶段重叠和告警提前量等评价方法；
5. 识别原方法的标签循环性和时间先验等局限。

后续主要实验应固定 VibFM：

```text
固定：振动 → VibFM → z_health

比较：
z_health → HI
z_health → K-means stages
z_health sequence → Change-Point + HMM
z_health sequence → GRU/Attention（必要时）
```

所以，是否完整复现作者的 AutoEncoder 应由“它是否有助于理解和验证 State Interpreter 后端”决定，而不是把完整 AE 复现当成进入 VibFM 工作的强制门槛。
