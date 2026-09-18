# XJTU-SY 65-feature / Feature LSTM最小移植结果

## 目标

在不读取真实bearing、不训练模型的前提下，把审计后的成熟工程基线移植为当前PyTorch仓库可测试的独立模块。

## 已实现

- `XJTUBearingFeatureExtractor`：双通道振动`[samples,2]`到有序65维float32特征；
- 37维时间域与28维频域特征；
- 三种XJTU-SY工况对应的BPFO、BPFI、BSF和FTF；
- `FeatureLSTM`：`[batch,steps,65]`到标量score和16维hidden representation；
- MIT来源与固定上游commit记录；
- NumPy和SciPy成为核心依赖。

## 与上游的差异

PyTorch LSTM分别保存input/recurrent bias，因此本实现有11,169个参数；等价Keras结构为11,041个参数。网络层次、hidden size和双向结构保持一致。本阶段没有移植上游RUL标签、curated onset、TensorFlow训练循环、MLflow或云端组件。

## 测试

定向测试：`10 passed`。

完整回归：第一次运行有117项通过、19项因Windows系统临时目录权限而无法建立fixture；将pytest basetemp指向已验证的仓库内目录后重新运行，结果为`136 passed`。

测试覆盖：

- 65维shape、顺序和float32类型；
- 常数信号无NaN/Inf；
- 水平/垂直通道频率不混淆；
- 三种工况的特征频率合法；
- 非法输入拒绝；
- Feature LSTM输出shape、参数数量与反向传播；
- 全仓库原有功能回归。

## 数据访问状态

只使用NumPy/Torch生成的人工信号。没有读取任何XJTU-SY CSV，尤其没有读取Bearing3_5。

## 结论

Phase A实现验证完成。代码接口和数值稳定性已满足进入真实数据特征提取前的门槛，但尚不能说明模型具有健康状态解释能力。

## 下一步

建立因果早期校准器和bearing内10步窗口构造器；随后先在允许的开发bearings上验证特征分布，再预注册Feature LSTM开发集LOBO训练协议。
