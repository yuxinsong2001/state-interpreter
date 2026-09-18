# TS2Vec官方论文与代码审计（2026-09-18）

## 来源

- 论文：Yue et al., *TS2Vec: Towards Universal Representation of Time Series*, AAAI 2022, DOI `10.1609/aaai.v36i8.20881`。
- 官方仓库：`zhihanyue/ts2vec`。
- 审计commit：`b0088e14a99706c05451316dc6db8d3da9351163`。
- 许可证：MIT，完整文本已保存至`licenses/TS2VEC_LICENSE.txt`。

## 方法核心

TS2Vec不是RUL模型，而是自监督时间序列表示学习框架：

1. 从同一序列随机产生两个具有重叠目标区间、但上下文范围不同的view；
2. 扩张卷积编码器为每个时间点生成contextual representation；
3. instance contrastive loss区分同一时间点来自哪个实例；
4. temporal contrastive loss区分同一实例中的不同时间点；
5. 在多层时间池化尺度上重复上述对比，形成hierarchical contrastive loss。

输入契约为`[instances, timestamps, features]`，输出为每个timestamp的representation。论文通过池化支持任意子序列粒度，官方代码也支持滑动窗口与causal inference。

## 与当前问题的匹配

当前65维工程特征天然构成`timestamps × 65 features`序列。TS2Vec不需要RUL标签，可以避免强迫健康稳定阶段线性下降；多尺度目标也比固定10步LSTM更适合不同生命周期长度。

但它不是自动的domain-invariant方法。官方instance contrastive loss把不同序列视作负样本，可能保留甚至增强bearing身份。当前项目已测得90.3%的identity probe，因此TS2Vec必须接受双重门槛：

- frozen representation上的LOBO健康排序是否改善；
- bearing身份准确率是否低于当前90.3%，至少不能进一步恶化。

## 不能直接照搬的部分

- 官方默认320维输出与项目期望的8/16维state不一致；第一轮应保留官方representation，再在训练bearing上拟合固定降维/读出，不能把320维直接称为state。
- 官方benchmark主要面向分类、预测和异常检测，没有PHM健康状态语义。
- 非causal整序列编码会看到未来，不可直接用于在线State Interpreter。
- 不同bearing长度不同，必须在读取验证集前锁定chunking、padding和causal inference规则。
- 不能用B3_4/B3_5选择表示维数、训练iterations或窗口长度。

## 已完成的最小移植

`src/state_interpreter/encoders/ts2vec.py`保留：

- SamePad dilated residual convolution；
- binomial timestamp masking；
- instance contrastive loss；
- temporal contrastive loss；
- hierarchical temporal pooling；
- overlapping context-view sampling。

实现已记录上游commit、版权和MIT许可证。人工数据测试覆盖输出shape、NaN padding、分层loss和反向传播；尚未读取真实bearing或训练TS2Vec。

## 下一步协议边界

下一步应在B3_1–B3_3内预注册TS2Vec开发实验，至少固定：

- chunk length与stride；
- encoder维数、depth与iterations；
- 三个随机种子；
- causal timestamp representation的生成方法；
- frozen linear/ordinal readout；
- identity与LOBO progress双重门槛。

只有开发门槛通过后才允许接触B3_4；B3_5继续作为最终holdout。
