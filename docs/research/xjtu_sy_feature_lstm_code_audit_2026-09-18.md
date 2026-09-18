# XJTU-SY Feature LSTM代码审计与复现边界

## Material Passport

- 类型：开源实现审计与复现实验规划
- 状态：ANALYZED（尚未运行训练）
- 审计日期：2026-09-18
- 上游仓库：[thfmn/xjtu-sy-bearing](https://github.com/thfmn/xjtu-sy-bearing)
- 固定上游版本：`7d7231c582961741bde629da6731e6c169d88785`
- 许可证：MIT
- 本地原始数据读取：无
- 受保护数据：未读取本地`Bearing3_5`文件

## 审计结论

该仓库适合作为**成熟工程基线来源**，但不能称为某篇论文的官方复现。仓库README明确说明Feature LSTM、LightGBM、CNN1D和CNN2D均为项目原创架构；因此本项目应把它称为“XJTU-SY专用开源工程基线”，而不是“文献中的State Interpreter”。

最值得复用的是三项设计：

1. 由物理和统计知识构成的65维双通道特征；
2. 基于设备早期健康段的per-bearing相对归一化；
3. 用连续10个测量组成窗口的小型Feature BiLSTM。

不应原样复用的是RUL标签、curated onset标签、同一留出bearing同时用于early stopping和最终报告的训练流程，以及依赖完整寿命比例的伪健康基线。

## 1. 65维特征的实际组成

### 时间域：37维

每通道18维，共36维，再加1维双通道Pearson相关系数：

- mean、standard deviation、variance；
- RMS、absolute peak、peak-to-peak；
- crest、shape、impulse、clearance factor；
- excess kurtosis、skewness；
- line integral、zero-crossing rate、Shannon entropy；
- 5%、50%、95%分位数；
- horizontal/vertical cross-correlation。

### 频域：28维

每通道14维：

- spectral centroid、bandwidth、85% rolloff、flatness；
- 0–1、1–3、3–6、6–12 kHz band power；
- dominant frequency、Welch PSD mean frequency；
- BPFO、BPFI、BSF、FTF附近band power。

频域特征使用XJTU-SY的25.6 kHz采样率、LDK UER204几何参数和各工况轴频率。实际实现为`37 + 28 = 65`。时间域模块部分docstring仍写“35/17”，与实际18维实现不一致，因此移植时必须用shape test确认65维，而不能只依赖注释。

## 2. Onset detection的实际方法

自动标注脚本先计算水平与垂直通道平均的kurtosis和RMS，再运行：

- Threshold detector：早期20%数据拟合均值和标准差，阈值为`mean + 2σ`，要求连续5点超过；
- CUSUM detector：`drift=0.5`、`threshold=5.0`；
- 选择优先级：kurtosis threshold → RMS threshold → kurtosis CUSUM → RMS CUSUM；若kurtosis在健康基线区内过早触发，则优先RMS。

仓库另外包含EWMA、Bayesian online change-point detection和ensemble，但推荐流程并未全部使用。

### 对当前项目的限制

“前20%寿命”需要知道整条轨迹长度，在真实在线部署时不可直接获得。curated onset labels也来自完整轨迹观察。因此：

- 可以在**忠实复现轨**中保留，以核对上游结果；
- 在**State Interpreter轨**中必须改为固定早期校准长度或严格因果的在线检测器；
- onset只能是phase/transition候选，不能自动等同于真实物理损伤起点。

## 3. Feature LSTM结构

输入为`[batch, 10, 65]`：

```text
10步×65维特征
→ BiLSTM(16)
→ Dropout(0.2)
→ Dense(16, ReLU)
→ Dense(1, linear)
```

训练配置为AdamW、学习率0.001、weight decay 0.0001、Huber loss、最多100 epochs、early stopping patience 7。

按Keras代码和设计指南，该双向模型为11,041个参数；README首页同时出现约5,793参数的说法，两者不一致。PyTorch的`nn.LSTM`分别保存input/recurrent bias，因此同结构移植后为11,169个参数，多出的128个均为偏置参数。两种实现的层级结构和hidden尺寸一致，参数差异已由单元测试固定。

虽然使用BiLSTM，但窗口只包含`t-9 … t`，没有使用`t+1`之后的数据；因此在窗口端点输出时仍可实现在线推理。不过它的原始输出是RUL，不是State Interpreter状态。

## 4. 评价协议与泄漏审计

### 可保留部分

- 窗口不跨bearing边界；
- LOBO按bearing拆分，而不是随机拆分测量文件；
- 每折构建新模型；
- 每个目标bearing可使用自身早期健康测量做相对校准，这与本项目现有15步校准思想一致。

### 必须修正部分

1. 上游脚本将LOBO留出bearing作为`validation_data`，同时用其`val_loss`做early stopping和best checkpoint选择，随后又在同一bearing报告指标。它是模型选择集，不是完全独立测试集。
2. full-life模式使用每个bearing总长度生成归一化RUL标签；该标签适合监督评价，但部署时总寿命未知。
3. post-onset模式加载15个bearing的curated onset标签，不能用于本项目最终保留集的无偏评价。
4. per-bearing归一化本身可在线，但“前20%”的边界不可在线；应改为固定前15步或只基于过去数据的动态校准。
5. 上游README的RMSE数值是仓库作者报告，尚未在本机复现，不能作为本项目已验证结果。

## 5. 与当前State Interpreter的接口关系

建议不要把上游Dense(1)的RUL输出直接作为最终state。第一版迁移接口应为：

```text
raw vibration [32768, 2]
→ 65维可解释特征
→ 固定早期校准/因果归一化
→ 10步Feature LSTM
→ hidden representation h_t
→ State head
→ [phase probability, relative progress, transition confidence]
```

对照实验同时保留一个`Dense(1)`监督健康进度输出。这样可以回答：成熟特征与时序模型是否解决当前AutoEncoder的一维轴不稳定，而不是只比较另一个RUL数字。

## 6. 依赖与实现决策

上游完整环境包含TensorFlow/Keras、MLflow、Google Cloud、Gradio、LightGBM等大量依赖。当前仓库使用PyTorch，因此不安装整个上游环境，也不复制全部仓库。

第一轮只移植MIT许可下的最小组件：

- NumPy/SciPy版65维特征定义；
- 因果健康基线归一化；
- PyTorch Feature LSTM；
- bearing级窗口与LOBO拆分；
- shape、数值稳定性和防泄漏测试。

这既保留成熟方法的核心归纳偏置，也避免引入TensorFlow、云平台和仪表盘依赖。

## 7. 下一轮实施门槛

### Phase A：实现验证，不训练

1. 新增`features/bearing_features.py`，输出固定65维及有序feature names；
2. 新增`encoders/feature_lstm.py`，输入`[B,10,65]`并返回hidden与score；
3. 对人工信号完成shape、finite value、常数信号和双通道测试；
4. 明确只允许B3_1–B3_3进入开发流程。

### Phase B：开发集LOBO

- 只用B3_1–B3_3轮换留出；
- checkpoint选择只能使用训练bearings中的inner validation，不使用LOBO test bearing；
- 多随机种子报告per-bearing均值与标准差；
- 与当前health-aware Encoder在相同Spearman、单调性和稳定性指标下比较。

### Phase C：锁定后验证

- 只有Phase B预注册门槛通过才读取B3_4；
- B3_5继续作为计算意义上的最终holdout；
- 不根据B3_4/B3_5重新选择特征、窗口或网络结构。

## 8. 盲测状态说明

本次没有读取本地B3_4/B3_5原始信号或运行任何推理。但是上游公开设计指南直接描述了包括Bearing3_2和Bearing2_5在内的个别轨迹特征，也包含数据集级结果。因此今后应准确表述为：B3_5仍是**未执行的计算holdout**，但公开资料可能提供数据集层面的先验，不能宣称研究人员对该数据集完全信息盲。

## 9. 决策

**GO，但仅限最小PyTorch移植和修正后的无泄漏协议。**

本次审计没有证明Feature LSTM一定优于当前方法；它证明该方法具有足够强、足够明确且可测试的归纳偏置，值得成为下一轮复现基线。

## 主要来源

- [上游README](https://github.com/thfmn/xjtu-sy-bearing/blob/main/README.md)
- [65维特征融合](https://github.com/thfmn/xjtu-sy-bearing/blob/main/src/features/fusion.py)
- [Feature LSTM实现](https://github.com/thfmn/xjtu-sy-bearing/blob/main/src/models/baselines/feature_lstm.py)
- [窗口与归一化](https://github.com/thfmn/xjtu-sy-bearing/blob/main/src/data/feature_windows.py)
- [Onset detectors](https://github.com/thfmn/xjtu-sy-bearing/blob/main/src/onset/detectors.py)
- [Feature LSTM训练脚本](https://github.com/thfmn/xjtu-sy-bearing/blob/main/scripts/11_train_feature_lstm.py)
- [项目依赖](https://github.com/thfmn/xjtu-sy-bearing/blob/main/pyproject.toml)
